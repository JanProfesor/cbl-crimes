import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from data_preparer_noscale import DataPreparerNoLeakage  
from tabnet_model import TabNetModel
from xgboost_model import XGBoostModel


def encode_categorical_columns(X_df: pd.DataFrame):
    """
    Encode any string or categorical columns in X_df to integer labels.
    Returns the encoded DataFrame and a dict of LabelEncoders per column.
    """
    X_df_encoded = X_df.copy()
    label_encoders = {}
    for col in X_df_encoded.columns:
        if (
            X_df_encoded[col].dtype == "object"
            or X_df_encoded[col].dtype.name == "category"
        ):
            le = LabelEncoder()
            X_df_encoded[col] = X_df_encoded[col].fillna('unknown')
            X_df_encoded[col] = le.fit_transform(X_df_encoded[col].astype(str))
            label_encoders[col] = le
    return X_df_encoded, label_encoders


def validate_data_quality(X, y, stage=""):
    print(f"\n=== Data Quality Check ({stage}) ===")
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    nan_count = X_df.isnull().sum().sum()
    inf_count = np.isinf(X_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaN values in features: {nan_count}")
    print(f"Inf values in features: {inf_count}")
    print(f"NaN values in target: {np.isnan(y).sum()}")
    print(f"Inf values in target: {np.isinf(y).sum()}")
    print(f"Target stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"Zero targets: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    if isinstance(X, np.ndarray):
        feature_vars = np.var(X, axis=0)
        zero_var_features = (feature_vars == 0).sum()
        print(f"Zero variance features: {zero_var_features}")
    return nan_count == 0 and inf_count == 0 and not np.isnan(y).any() and not np.isinf(y).any()


def fix_data_issues(X, y):
    X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
    X_df = X_df.fillna(X_df.median())
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(X_df.median())
    y_fixed = np.nan_to_num(y, nan=np.median(y[~np.isnan(y)]) if not np.isnan(y).all() else 0)
    y_fixed = np.clip(y_fixed, -1e6, 1e6)
    return X_df.values, y_fixed


def main():
    csv_path   = "ward_london.csv"
    target_col = "burglary_count"
    ensemble_w = 0.48
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

    try:
        preparer = DataPreparerNoLeakage(csv_path, target_col)
        df_train_full, df_test_full, train_end_date = preparer.preprocess_split_aware()
        print(f"Train period: up to {train_end_date}")
        print(f"Training samples: {len(df_train_full)}")
        print(f"Test samples: {len(df_test_full)}")

        X_train_df, y_train = preparer.prepare_features(df_train_full)
        X_test_df,  y_test  = preparer.prepare_features(df_test_full)
        if X_test_df.empty:
            print("WARNING: No test data available after feature engineering!")
            return

        y_train = np.log1p(y_train)
        y_test  = np.log1p(y_test)

        to_drop = ["ward_code","ward_code_orig","year_orig","month_orig","actual"]
        X_train_clean = X_train_df.drop(columns=to_drop, errors="ignore")
        X_test_clean  = X_test_df.drop(columns=to_drop, errors="ignore")

        cat_cols = [c for c in X_train_clean.columns if X_train_clean[c].dtype in ["object","category"]]
        if cat_cols:
            X_train_encoded, label_encoders = encode_categorical_columns(X_train_clean)
            X_test_encoded  = X_test_clean.copy()
            for col in cat_cols:
                le = label_encoders[col]
                X_test_encoded[col] = X_test_encoded[col].fillna('unknown')
                mask = X_test_encoded[col].astype(str).isin(le.classes_)
                X_test_encoded.loc[mask, col] = le.transform(X_test_encoded.loc[mask, col].astype(str))
                X_test_encoded.loc[~mask, col] = -1
        else:
            X_train_encoded, X_test_encoded, label_encoders = X_train_clean.copy(), X_test_clean.copy(), {}

        cat_idxs_new = [X_train_encoded.columns.get_loc(c) for c in cat_cols]
        cat_dims_new = [len(label_encoders[c].classes_)+2 for c in cat_cols]

        num_cols = [c for c in X_train_encoded.columns if c not in cat_cols]
        scaler = StandardScaler()
        X_train_scaled = X_train_encoded.copy()
        X_test_scaled  = X_test_encoded.copy()
        if num_cols:
            X_train_scaled[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
            X_test_scaled[num_cols]  = scaler.transform(X_test_encoded[num_cols])

        X_train_values, y_train = fix_data_issues(X_train_scaled.values, y_train)
        X_test_values,  y_test  = fix_data_issues(X_test_scaled.values,  y_test)

        if not validate_data_quality(X_train_values,y_train,"Training"):
            print("ERROR: Training data has quality issues!"); return
        if not validate_data_quality(X_test_values,y_test,"Test"):
            print("ERROR: Test data has quality issues!"); return

        print(f"Features={X_train_values.shape[1]}, Cats={len(cat_cols)}, Num={len(num_cols)}")

        # ─── TIME-AWARE CROSS-VALIDATION ─────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=3, test_size=300)
        tabnet_scores = {'rmse':[], 'mae':[], 'r2':[]}
        xgb_scores   = {'rmse':[], 'mae':[], 'r2':[]}
        ens_scores   = {'rmse':[], 'mae':[], 'r2':[]}

        try:
            with open("runs/best_run_yet/best_tabnet_params.json") as f:
                tabnet_params = json.load(f)
            with open("runs/best_run_yet/best_xgb_params.json") as f:
                xgb_params = json.load(f)
        except:
            tabnet_params = {'n_d':644,'n_a':64,'n_steps':7,'gamma':2,'lr':0.01,'batch_size':512,
                              'virtual_batch_size':256,'max_epochs':200,'patience':20,'mask_type':'entmax',
                              'lambda_sparse':1e-3,'weight_decay':1e-5}
            xgb_params = {'n_estimators':1000,'max_depth':8,'learning_rate':0.05,'subsample':0.8,
                          'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':10,'random_state':42,
                          'tree_method':'gpu_hist','predictor':'gpu_predictor'}

        print("\nRunning time-aware cross-validation...")
        valid_folds=0
        for fold,(tr_idx,val_idx) in enumerate(tscv.split(X_train_values[:15000]),1):
            X_tr, X_val = X_train_values[tr_idx], X_train_values[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            if len(X_tr)<50 or len(X_val)<10: continue
            # internal holdout
            X_tr_sub,X_val_sub,y_tr_sub,y_val_sub = train_test_split(X_tr,y_tr,test_size=0.1,shuffle=False)
            # TabNet
            tabnet=TabNetModel(cat_idxs_new,cat_dims_new,device)
            tabnet.train(X_tr_sub,y_tr_sub,tabnet_params,X_val=X_val_sub,y_val=y_val_sub)
            pred_tab=np.expm1(tabnet.predict(X_val))
            # XGBoost
            xgbm=XGBoostModel(device)
            xgbm.train(X_tr_sub,y_tr_sub,xgb_params,eval_set=(X_val_sub,y_val_sub))
            pred_xgb=np.expm1(xgbm.predict(X_val))
            pred_ens=ensemble_w*pred_tab+(1-ensemble_w)*pred_xgb
            yv=np.expm1(y_val)
            rmse=lambda yt,yp: np.sqrt(mean_squared_error(yt,yp))
            tabnet_scores['rmse'].append(rmse(yv,pred_tab)); xgb_scores['rmse'].append(rmse(yv,pred_xgb)); ens_scores['rmse'].append(rmse(yv,pred_ens))
            valid_folds+=1
        print(f"Done CV: {valid_folds} folds")

        # ─── FINAL HOLDOUT EVALUATION ──────────────────────────────────────────
        print("\nTraining final models on full training set...")
        # TabNet final
        X_tr_full,X_val_full,y_tr_full,y_val_full= train_test_split(X_train_values,y_train,test_size=0.1,shuffle=False)
        final_tabnet_params = tabnet_params.copy()
        final_tabnet_params.update({'lr':0.005,'batch_size':1024,'virtual_batch_size':512,'max_epochs':100,'patience':10,'lambda_sparse':5e-3})
        tabnet_final=TabNetModel(cat_idxs_new,cat_dims_new,device)
        tabnet_final.train(X_tr_full,y_tr_full,final_tabnet_params,X_val=X_val_full,y_val=y_val_full)
        pred_tab_test=np.expm1(tabnet_final.predict(X_test_values))
        # XGBoost final
        xgb_final=XGBoostModel(device)
        xgb_final.train(X_train_values,y_train,xgb_params)
        pred_xgb_test=np.expm1(xgb_final.predict(X_test_values))
        pred_ens_test=ensemble_w*pred_tab_test+(1-ensemble_w)*pred_xgb_test

        y_test_orig=np.round(np.expm1(y_test)).astype(int)
        def safe_final_metrics(y_true,y_pred):
            return (np.sqrt(mean_squared_error(y_true,y_pred)),
                    mean_absolute_error(y_true,y_pred),
                    r2_score(y_true,y_pred))
        rmse_tab,mae_tab,r2_tab=safe_final_metrics(y_test_orig,pred_tab_test)
        rmse_xgb,mae_xgb,r2_xgb=safe_final_metrics(y_test_orig,pred_xgb_test)
        rmse_ens,mae_ens,r2_ens=safe_final_metrics(y_test_orig,pred_ens_test)
        print(f"\n=== Final Holdout Results ===")
        print(f"TabNet   | RMSE: {rmse_tab:.4f} | MAE: {mae_tab:.4f} | R²: {r2_tab:.4f}")
        print(f"XGBoost  | RMSE: {rmse_xgb:.4f} | MAE: {mae_xgb:.4f} | R²: {r2_xgb:.4f}")
        print(f"Ensemble | RMSE: {rmse_ens:.4f} | MAE: {mae_ens:.4f} | R²: {r2_ens:.4f}")
        # save
        holdout_df=pd.DataFrame({"model":["TabNet","XGBoost","Ensemble"],
                                 "rmse":[rmse_tab,rmse_xgb,rmse_ens],
                                 "mae":[mae_tab,mae_xgb,mae_ens],
                                 "r2":[r2_tab,r2_xgb,r2_ens]})
        holdout_df.to_csv("holdout_metrics_fixed.csv",index=False)
        if not df_test_full.empty:
            results_df=pd.DataFrame({"ward_code":df_test_full['ward_code_orig'],"year":df_test_full['year_orig'],"month":df_test_full['month_orig'],
                                      "actual":y_test_orig,"pred_tabnet":pred_tab_test,"pred_xgboost":pred_xgb_test,"pred_ensemble":pred_ens_test})
            results_df.to_csv("test_predictions_fixed.csv",index=False)
        print("Results saved.")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback; traceback.print_exc()

if __name__=="__main__":
    main()

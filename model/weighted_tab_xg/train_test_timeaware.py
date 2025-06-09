import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize_scalar
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


def optimize_ensemble_weights(pred_tab, pred_xgb, y_true):
    """
    Optimize ensemble weights to minimize underprediction
    """
    def objective(weight):
        ensemble_pred = weight * pred_tab + (1 - weight) * pred_xgb
        # Custom metric that penalizes underprediction more
        errors = y_true - ensemble_pred
        underpred_penalty = np.where(errors > 0, errors * 2, errors)  # 2x penalty for underprediction
        return np.mean(np.abs(underpred_penalty))
    
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return result.x


def simple_bias_correction(predictions, y_true_validation, pred_validation):
    """
    Simple multiplicative bias correction
    """
    correction_factor = np.mean(y_true_validation) / np.mean(pred_validation) if np.mean(pred_validation) > 0 else 1.0
    return predictions * correction_factor, correction_factor


def main():
    csv_path   = "ward_london.csv"
    target_col = "burglary_count"
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

        y_train_log = np.log1p(y_train)
        y_test_log  = np.log1p(y_test)

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

        X_train_values, y_train_log = fix_data_issues(X_train_scaled.values, y_train_log)
        X_test_values,  y_test_log  = fix_data_issues(X_test_scaled.values,  y_test_log)

        if not validate_data_quality(X_train_values, y_train_log, "Training"):
            print("ERROR: Training data has quality issues!"); return
        if not validate_data_quality(X_test_values, y_test_log, "Test"):
            print("ERROR: Test data has quality issues!"); return

        print(f"Features={X_train_values.shape[1]}, Cats={len(cat_cols)}, Num={len(num_cols)}")

        # Load or set parameters with reduced regularization
        try:
            with open("runs/best_run_yet/best_tabnet_params.json") as f:
                tabnet_params = json.load(f)
            with open("runs/best_run_yet/best_xgb_params.json") as f:
                xgb_params = json.load(f)
        except:
            # Reduced regularization to address underprediction
            tabnet_params = {'n_d':644,'n_a':64,'n_steps':7,'gamma':1.5,'lr':0.015,'batch_size':512,
                              'virtual_batch_size':256,'max_epochs':200,'patience':20,'mask_type':'entmax',
                              'lambda_sparse':5e-4,'weight_decay':5e-6}
            xgb_params = {'n_estimators':1200,'max_depth':9,'learning_rate':0.03,'subsample':0.85,
                          'colsample_bytree':0.85,'reg_alpha':0.05,'reg_lambda':5,'random_state':42,
                          'tree_method':'gpu_hist','predictor':'gpu_predictor'}

        # ─── TIME-AWARE CROSS-VALIDATION WITH ENSEMBLE OPTIMIZATION ─────────────
        tscv = TimeSeriesSplit(n_splits=3, test_size=300)
        cv_tabnet_preds = []
        cv_xgb_preds = []
        cv_y_true = []

        print("\nRunning time-aware cross-validation...")
        valid_folds = 0
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_values[:15000]), 1):
            X_tr, X_val = X_train_values[tr_idx], X_train_values[val_idx]
            y_tr, y_val = y_train_log[tr_idx], y_train_log[val_idx]
            if len(X_tr) < 50 or len(X_val) < 10: 
                continue
                
            # Internal holdout
            X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(X_tr, y_tr, test_size=0.1, shuffle=False)
            
            # TabNet
            tabnet = TabNetModel(cat_idxs_new, cat_dims_new, device)
            tabnet.train(X_tr_sub, y_tr_sub, tabnet_params, X_val=X_val_sub, y_val=y_val_sub)
            pred_tab = np.expm1(tabnet.predict(X_val))
            
            # XGBoost
            xgbm = XGBoostModel(device)
            xgbm.train(X_tr_sub, y_tr_sub, xgb_params, eval_set=(X_val_sub, y_val_sub))
            pred_xgb = np.expm1(xgbm.predict(X_val))
            
            # Store for ensemble optimization
            cv_tabnet_preds.append(pred_tab)
            cv_xgb_preds.append(pred_xgb)
            cv_y_true.append(np.expm1(y_val))
            
            valid_folds += 1

        print(f"Done CV: {valid_folds} folds")

        # Optimize ensemble weights
        if valid_folds > 0:
            cv_tabnet_all = np.concatenate(cv_tabnet_preds)
            cv_xgb_all = np.concatenate(cv_xgb_preds)
            cv_y_all = np.concatenate(cv_y_true)
            
            optimal_weight = optimize_ensemble_weights(cv_tabnet_all, cv_xgb_all, cv_y_all)
            print(f"Optimal ensemble weight (TabNet): {optimal_weight:.4f}")
        else:
            optimal_weight = 0.48  # Default

        # ─── FINAL TRAINING AND PREDICTION ──────────────────────────────────────
        print("\nTraining final models on full training set...")
        
        # TabNet final with reduced regularization
        X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
            X_train_values, y_train_log, test_size=0.1, shuffle=False
        )
        final_tabnet_params = tabnet_params.copy()
        final_tabnet_params.update({
            'lr': 0.008, 'batch_size': 1024, 'virtual_batch_size': 512, 
            'max_epochs': 120, 'patience': 15, 'lambda_sparse': 2e-4
        })
        
        tabnet_final = TabNetModel(cat_idxs_new, cat_dims_new, device)
        tabnet_final.train(X_tr_full, y_tr_full, final_tabnet_params, X_val=X_val_full, y_val=y_val_full)
        pred_tab_test = np.expm1(tabnet_final.predict(X_test_values))
        
        # XGBoost final
        xgb_final = XGBoostModel(device)
        xgb_final.train(X_train_values, y_train_log, xgb_params)
        pred_xgb_test = np.expm1(xgb_final.predict(X_test_values))
        
        # Apply simple bias correction
        val_pred_tab = np.expm1(tabnet_final.predict(X_val_full))
        val_pred_xgb = np.expm1(xgb_final.predict(X_val_full))
        val_y_true = np.expm1(y_val_full)
        
        pred_tab_test, tab_correction = simple_bias_correction(pred_tab_test, val_y_true, val_pred_tab)
        pred_xgb_test, xgb_correction = simple_bias_correction(pred_xgb_test, val_y_true, val_pred_xgb)
        
        print(f"Applied bias correction - TabNet: {tab_correction:.4f}, XGBoost: {xgb_correction:.4f}")
        
        # Create ensemble with optimized weights
        pred_ens_test = optimal_weight * pred_tab_test + (1 - optimal_weight) * pred_xgb_test
        
        # Apply additional 5% upward adjustment to reduce underprediction
        pred_ens_test *= 1.05
        print("Applied 5% upward adjustment to ensemble predictions")

        # Final evaluation
        y_test_final = np.round(np.expm1(y_test_log)).astype(int)
        
        def comprehensive_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Underprediction analysis
            errors = y_true - y_pred
            underpred_count = (errors > 0).sum()
            underpred_pct = underpred_count / len(errors) * 100
            mean_underpred = errors[errors > 0].mean() if underpred_count > 0 else 0
            
            # Mean prediction vs actual
            pred_mean = y_pred.mean()
            actual_mean = y_true.mean()
            bias = (pred_mean - actual_mean) / actual_mean * 100
            
            return {
                'rmse': rmse, 'mae': mae, 'r2': r2,
                'underpred_pct': underpred_pct, 'mean_underpred': mean_underpred,
                'bias_pct': bias, 'pred_mean': pred_mean, 'actual_mean': actual_mean
            }
        
        metrics_tab = comprehensive_metrics(y_test_final, pred_tab_test)
        metrics_xgb = comprehensive_metrics(y_test_final, pred_xgb_test)
        metrics_ens = comprehensive_metrics(y_test_final, pred_ens_test)
        
        print(f"\n=== Quick Fix Results ===")
        print(f"TabNet   | RMSE: {metrics_tab['rmse']:.4f} | MAE: {metrics_tab['mae']:.4f} | R²: {metrics_tab['r2']:.4f}")
        print(f"         | Underpred: {metrics_tab['underpred_pct']:.1f}% | Bias: {metrics_tab['bias_pct']:.1f}%")
        print(f"XGBoost  | RMSE: {metrics_xgb['rmse']:.4f} | MAE: {metrics_xgb['mae']:.4f} | R²: {metrics_xgb['r2']:.4f}")
        print(f"         | Underpred: {metrics_xgb['underpred_pct']:.1f}% | Bias: {metrics_xgb['bias_pct']:.1f}%")
        print(f"Ensemble | RMSE: {metrics_ens['rmse']:.4f} | MAE: {metrics_ens['mae']:.4f} | R²: {metrics_ens['r2']:.4f}")
        print(f"         | Underpred: {metrics_ens['underpred_pct']:.1f}% | Bias: {metrics_ens['bias_pct']:.1f}%")
        print(f"         | Pred Mean: {metrics_ens['pred_mean']:.2f} | Actual Mean: {metrics_ens['actual_mean']:.2f}")
        
        # Save results
        holdout_df = pd.DataFrame({
            "model": ["TabNet", "XGBoost", "Ensemble"],
            "rmse": [metrics_tab['rmse'], metrics_xgb['rmse'], metrics_ens['rmse']],
            "mae": [metrics_tab['mae'], metrics_xgb['mae'], metrics_ens['mae']],
            "r2": [metrics_tab['r2'], metrics_xgb['r2'], metrics_ens['r2']],
            "underpred_pct": [metrics_tab['underpred_pct'], metrics_xgb['underpred_pct'], metrics_ens['underpred_pct']],
            "bias_pct": [metrics_tab['bias_pct'], metrics_xgb['bias_pct'], metrics_ens['bias_pct']]
        })
        holdout_df.to_csv("quick_fix_holdout_metrics.csv", index=False)
        
        if not df_test_full.empty:
            results_df = pd.DataFrame({
                "ward_code": df_test_full['ward_code_orig'],
                "year": df_test_full['year_orig'],
                "month": df_test_full['month_orig'],
                "actual": y_test_final,
                "pred_tabnet": pred_tab_test,
                "pred_xgboost": pred_xgb_test,
                "pred_ensemble": pred_ens_test,
                "error": y_test_final - pred_ens_test,
                "abs_error": np.abs(y_test_final - pred_ens_test)
            })
            results_df.to_csv("quick_fix_test_predictions.csv", index=False)
        
        print("Quick fix results saved.")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
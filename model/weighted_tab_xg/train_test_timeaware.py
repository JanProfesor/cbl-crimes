import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
            # Handle NaN values before encoding
            X_df_encoded[col] = X_df_encoded[col].fillna('unknown')
            X_df_encoded[col] = le.fit_transform(X_df_encoded[col].astype(str))
            label_encoders[col] = le
    return X_df_encoded, label_encoders

def validate_data_quality(X, y, stage=""):
    """Validate data for common issues that cause training problems"""
    print(f"\n=== Data Quality Check ({stage}) ===")
    
    # Check for NaN/inf values
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    nan_count = X_df.isnull().sum().sum()
    inf_count = np.isinf(X_df.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"NaN values in features: {nan_count}")
    print(f"Inf values in features: {inf_count}")
    print(f"NaN values in target: {np.isnan(y).sum()}")
    print(f"Inf values in target: {np.isinf(y).sum()}")
    
    # Check target distribution
    print(f"Target stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"Zero targets: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    
    # Check feature variance
    if isinstance(X, np.ndarray):
        feature_vars = np.var(X, axis=0)
        zero_var_features = (feature_vars == 0).sum()
        print(f"Zero variance features: {zero_var_features}")
    
    return nan_count == 0 and inf_count == 0 and not np.isnan(y).any() and not np.isinf(y).any()

def fix_data_issues(X, y):
    """Fix common data issues"""
    # Convert to DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # Handle NaN values
    X_df = X_df.fillna(X_df.median())
    
    # Handle infinite values
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median())
    
    # Fix target issues
    y_fixed = np.nan_to_num(y, nan=np.median(y[~np.isnan(y)]) if not np.isnan(y).all() else 0)
    y_fixed = np.clip(y_fixed, -1e6, 1e6)  # Clip extreme values
    
    return X_df.values, y_fixed

def main():
    # ─── CONFIGURATION ───────────────────────────────────────────────────────────
    csv_path   = "processed/final_dataset_residential_burglary_reordered.csv"
    target_col = "burglary_count"
    ensemble_w = 0.48
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

    try:
        # 1) LOAD & PREPROCESS with proper temporal splitting to avoid leakage
        preparer = DataPreparerNoLeakage(csv_path, target_col)
        df_train_full, df_test_full, train_end_date = preparer.preprocess_split_aware()
        
        print(f"Train period: up to {train_end_date}")
        print(f"Training samples: {len(df_train_full)}")
        print(f"Test samples: {len(df_test_full)}")
        
        # 2) Extract features and targets
        X_train_df, y_train = preparer.prepare_features(df_train_full)
        X_test_df, y_test   = preparer.prepare_features(df_test_full)

        if X_test_df.empty:
            print("WARNING: No test data available after feature engineering!")
            return
        
        # Log-transform the target for both train and test 
        y_train = np.log1p(y_train)
        y_test  = np.log1p(y_test)

        # 3) Remove leakage-prone columns
        to_drop = ["ward_code", "ward_code_orig", "year_orig", "month_orig", "actual"]
        X_train_clean = X_train_df.drop(columns=to_drop, errors="ignore")
        X_test_clean  = X_test_df.drop(columns=to_drop, errors="ignore")
        
        # 4) Handle categorical columns
        cat_cols = [
            col for col in X_train_clean.columns
            if X_train_clean[col].dtype == "object" or X_train_clean[col].dtype.name == "category"
        ]
        
        # Encode categoricals (fit on train, transform both)
        if cat_cols:
            print(f"Encoding categorical columns: {cat_cols}")
            X_train_encoded, label_encoders = encode_categorical_columns(X_train_clean)
            X_test_encoded = X_test_clean.copy()
            for col in cat_cols:
                if col in X_test_encoded.columns:
                    # Handle unseen categories more robustly
                    le = label_encoders[col]
                    X_test_encoded[col] = X_test_encoded[col].fillna('unknown')
                    
                    # Transform known categories, assign -1 to unknown
                    mask = X_test_encoded[col].astype(str).isin(le.classes_)
                    X_test_encoded.loc[mask, col] = le.transform(X_test_encoded.loc[mask, col].astype(str))
                    X_test_encoded.loc[~mask, col] = -1  # Unknown category marker
        else:
            X_train_encoded = X_train_clean.copy()
            X_test_encoded  = X_test_clean.copy()
            label_encoders  = {}

        # 5) Prepare categorical indices & dimensions for TabNet
        cat_idxs_new = [X_train_encoded.columns.get_loc(col) for col in cat_cols] if cat_cols else []
        cat_dims_new = [len(label_encoders[col].classes_) + 2 for col in cat_cols] if cat_cols else []  # +2 for unknown and padding

        # 6) Scale numerical features (fit on train only!)
        all_cols  = list(X_train_encoded.columns)
        num_cols  = [c for c in all_cols if c not in cat_cols]
        
        scaler = StandardScaler()
        X_train_scaled = X_train_encoded.copy()
        X_test_scaled  = X_test_encoded.copy()
        
        if num_cols:
            print(f"Scaling numerical columns: {len(num_cols)} features")
            X_train_scaled[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
            X_test_scaled[num_cols]  = scaler.transform(X_test_encoded[num_cols])
        
        # Convert to numpy arrays and fix data issues
        X_train_values, y_train = fix_data_issues(X_train_scaled.values, y_train)
        X_test_values,  y_test  = fix_data_issues(X_test_scaled.values,  y_test)
        
        # Validate data quality
        if not validate_data_quality(X_train_values, y_train, "Training"):
            print("ERROR: Training data has quality issues!")
            return
            
        if not validate_data_quality(X_test_values, y_test, "Test"):
            print("ERROR: Test data has quality issues!")
            return
        
        print(f"Features after cleaning: {X_train_values.shape[1]}")
        print(f"Categorical features: {len(cat_cols)}")
        print(f"Numerical features: {len(num_cols)}")

        # ─── TIME-AWARE CROSS-VALIDATION ───────────────────────────────────────────
        
        # Use a reasonable subset for CV
        cv_samples = min(len(X_train_values), 15000)
        X_cv = X_train_values[:cv_samples]
        y_cv = y_train[:cv_samples]
        
        # Check if we have enough data for time series split
        if len(X_cv) < 100:
            print("WARNING: Not enough data for cross-validation, skipping CV...")
            cv_metrics = None
        else:
            tscv = TimeSeriesSplit(n_splits=3, test_size=max(100, len(X_cv)//10))
  
            tabnet_scores = {'rmse': [], 'mae': [], 'r2': []}
            xgb_scores   = {'rmse': [], 'mae': [], 'r2': []}
            ens_scores   = {'rmse': [], 'mae': [], 'r2': []}

            # Load hyperparameters with better defaults
            try:
                with open("runs/best_run_yet/best_tabnet_params.json") as f:
                    tabnet_params = json.load(f)
                with open("runs/best_run_yet/best_xgb_params.json") as f:
                    xgb_params = json.load(f)
            except FileNotFoundError:
                print("Using improved default parameters")
                tabnet_params = {
                    'n_d': 16, 'n_a': 16, 'n_steps': 3, 'gamma': 1.3,
                    'lr': 0.02, 'batch_size': 512, 'virtual_batch_size': 256,
                    'max_epochs': 50, 'patience': 10
                }
                xgb_params = {
                    'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                    'subsample': 0.8, 'colsample_bytree': 0.8,
                    'random_state': 42, 'verbosity': 0
                }

            print("\nRunning time-aware cross-validation...")
            valid_folds = 0

            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_cv), start=1):
                try:
                    X_tr, X_val = X_cv[tr_idx], X_cv[val_idx]
                    y_tr, y_val = y_cv[tr_idx], y_cv[val_idx]
        
                    # Skip fold if insufficient data
                    if len(X_tr) < 50 or len(X_val) < 10:
                        print(f"Fold {fold}: Insufficient data, skipping...")
                        continue
        
                    print(f"Fold {fold}: Train={len(X_tr)}, Val={len(X_val)}")

                    # ─── TRAIN TabNet on log‐transformed target ────────────────────────────
                    tabnet = TabNetModel(cat_idxs_new, cat_dims_new, device)
                    tabnet.train(X_tr, y_tr, tabnet_params)

                    # TabNet’s predict() returns predictions in log-space
                    pred_tab_log = tabnet.predict(X_val).astype(float)
                    # Convert back to original burglary‐count scale
                    pred_tab = np.expm1(pred_tab_log)

                    # ─── TRAIN XGBoost on log‐transformed target ─────────────────────────
                    xgbm = XGBoostModel(device)
                    xgbm.train(X_tr, y_tr, xgb_params)

                    # XGBoost’s predict() also returns predictions in log-space
                    pred_xgb_log = xgbm.predict(X_val).astype(float)
                    # Convert back to original scale
                    pred_xgb = np.expm1(pred_xgb_log)

                    # Handle any NaNs or infinities (just in case)
                    # We use expm1(y_val.mean()) as a fallback “typical burglary count”
                    pred_tab = np.nan_to_num(pred_tab, nan=np.expm1(y_val.mean()))
                    pred_xgb = np.nan_to_num(pred_xgb, nan=np.expm1(y_val.mean()))

                    # ─── ENSEMBLE on original scale ───────────────────────────────────────
                    pred_ens = ensemble_w * pred_tab + (1 - ensemble_w) * pred_xgb

                    # ─── Inverse‐log the “true” y_val before computing metrics ────────────
                    y_val_orig = np.expm1(y_val)

                    # Compute metrics with error handling
                    def safe_metrics(y_true, y_pred):
                        try:
                            y_true = np.nan_to_num(y_true)
                            y_pred = np.nan_to_num(y_pred)
                            
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae  = mean_absolute_error(y_true, y_pred)
                            r2   = r2_score(y_true, y_pred)
                            
                            # Cap extreme values
                            rmse = min(rmse, 1000)
                            mae  = min(mae, 1000)
                            r2   = max(r2,   -10)  
                            
                            return rmse, mae, r2
                        except Exception as e:
                            print(f"Metrics calculation failed: {e}")
                            return 999, 999, -999

                    # ─── Compute metrics on the ORIGINAL scale ───────────────────────────
                    rmse_t, mae_t, r2_t = safe_metrics(y_val_orig, pred_tab)
                    rmse_x, mae_x, r2_x = safe_metrics(y_val_orig, pred_xgb)
                    rmse_e, mae_e, r2_e = safe_metrics(y_val_orig, pred_ens)
                    # ─── “TRAIN & PREDICT” BLOCK END ─────────────────────────────────────

                    tabnet_scores['rmse'].append(rmse_t)
                    tabnet_scores['mae'].append(mae_t)
                    tabnet_scores['r2'].append(r2_t)
                    
                    xgb_scores['rmse'].append(rmse_x)
                    xgb_scores['mae'].append(mae_x)
                    xgb_scores['r2'].append(r2_x)
                    
                    ens_scores['rmse'].append(rmse_e)
                    ens_scores['mae'].append(mae_e)
                    ens_scores['r2'].append(r2_e)

                    print(f"Fold {fold} | TabNet RMSE: {rmse_t:.4f}, XGB RMSE: {rmse_x:.4f}, Ens RMSE: {rmse_e:.4f}")
                    valid_folds += 1
        
                except Exception as e:
                    print(f"Fold {fold} failed with error: {e}")
                    continue

            # Compute CV metrics if at least one fold succeeded
            if valid_folds > 0:
                cv_metrics = {
                    "TabNet": {
                        "rmse_mean": float(np.mean(tabnet_scores['rmse'])), 
                        "rmse_std":  float(np.std(tabnet_scores['rmse'])),
                        "mae_mean":  float(np.mean(tabnet_scores['mae'])), 
                        "mae_std":   float(np.std(tabnet_scores['mae'])),
                        "r2_mean":   float(np.mean(tabnet_scores['r2'])), 
                        "r2_std":    float(np.std(tabnet_scores['r2'])),
                    },
                    "XGBoost": {
                        "rmse_mean": float(np.mean(xgb_scores['rmse'])), 
                        "rmse_std":  float(np.std(xgb_scores['rmse'])),
                        "mae_mean":  float(np.mean(xgb_scores['mae'])), 
                        "mae_std":   float(np.std(xgb_scores['mae'])),
                        "r2_mean":   float(np.mean(xgb_scores['r2'])), 
                        "r2_std":    float(np.std(xgb_scores['r2'])),
                    },
                    "Ensemble": {
                        "rmse_mean": float(np.mean(ens_scores['rmse'])), 
                        "rmse_std":  float(np.std(ens_scores['rmse'])),
                        "mae_mean":  float(np.mean(ens_scores['mae'])), 
                        "mae_std":   float(np.std(ens_scores['mae'])),
                        "r2_mean":   float(np.mean(ens_scores['r2'])), 
                        "r2_std":    float(np.std(ens_scores['r2'])),
                    },
                }

                print(f"\n=== CV Results ({valid_folds} valid folds) ===")
                for model_name in ["TabNet", "XGBoost", "Ensemble"]:
                    m = cv_metrics[model_name]
                    print(
                        f"{model_name:8s} | "
                        f"RMSE = {m['rmse_mean']:.4f} ± {m['rmse_std']:.4f} | "
                        f"MAE = {m['mae_mean']:.4f} ± {m['mae_std']:.4f} | "
                        f"R² = {m['r2_mean']:.4f} ± {m['r2_std']:.4f}"
                    )
            else:
                cv_metrics = None

        # ─── After CV: Save CV metrics if available ─────────────────────────────
        if cv_metrics is not None:
            with open("cv_metrics_fixed.json", "w") as f:
                json.dump(cv_metrics, f, indent=4)

        # ─── FINAL HOLDOUT EVALUATION ────────────────────────────────────────────
        print("\nTraining final models on full training set...")

        # Load final hyperparameters (fallback to defaults if missing)
        try:
            with open("runs/best_run_yet/best_tabnet_params.json") as f:
                final_tabnet_params = json.load(f)
            with open("runs/best_run_yet/best_xgb_params.json") as f:
                final_xgb_params = json.load(f)
        except FileNotFoundError:
            final_tabnet_params = {
                'n_d': 32, 'n_a': 32, 'n_steps': 5, 'gamma': 1.3,
                'lr': 0.02, 'batch_size': 1024, 'virtual_batch_size': 512,
                'max_epochs': 100, 'patience': 15
            }
            final_xgb_params = {
                'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'random_state': 42, 'verbosity': 0
            }

        # Train final TabNet on the full (log‐transformed) training set
        try:
            print("Training final TabNet...")
            tabnet_final = TabNetModel(cat_idxs_new, cat_dims_new, device)
            tabnet_final.train(X_train_values, y_train, final_tabnet_params)

            # Predict (log-space), then invert
            pred_tab_test_log = tabnet_final.predict(X_test_values).astype(float)
            pred_tab_test     = np.expm1(pred_tab_test_log)
            # Handle NaNs/Infs
            pred_tab_test     = np.nan_to_num(pred_tab_test, nan=np.expm1(y_test.mean()))
        except Exception as e:
            print(f"Final TabNet training failed: {e}")
            # Fallback to mean of original counts
            pred_tab_test = np.full(len(y_test), np.expm1(y_test.mean()))

        # Train final XGBoost on the full (log‐transformed) training set
        try:
            print("Training final XGBoost...")
            xgb_final = XGBoostModel(device)
            xgb_final.train(X_train_values, y_train, final_xgb_params)

            # Predict (log-space), then invert
            pred_xgb_test_log = xgb_final.predict(X_test_values).astype(float)
            pred_xgb_test     = np.expm1(pred_xgb_test_log)
            # Handle NaNs/Infs
            pred_xgb_test     = np.nan_to_num(pred_xgb_test, nan=np.expm1(y_test.mean()))
        except Exception as e:
            print(f"Final XGBoost training failed: {e}")
            # Fallback to mean of original counts
            pred_xgb_test = np.full(len(y_test), np.expm1(y_test.mean()))

        # Ensemble on original-count scale
        pred_ens_test = ensemble_w * pred_tab_test + (1 - ensemble_w) * pred_xgb_test

        # Invert the “true” test targets
        y_test_orig = np.expm1(y_test)

        # Compute final holdout metrics
        def safe_final_metrics(y_true, y_pred, model_name):
            try:
                y_true = np.nan_to_num(y_true)
                y_pred = np.nan_to_num(y_pred)
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae  = mean_absolute_error(y_true, y_pred)
                r2   = r2_score(y_true, y_pred)
                
                return rmse, mae, r2
            except Exception as e:
                print(f"Final metrics calculation failed for {model_name}: {e}")
                return 999, 999, -999

        rmse_tab_test, mae_tab_test, r2_tab_test = safe_final_metrics(
            y_test_orig, pred_tab_test, "TabNet"
        )
        rmse_xgb_test, mae_xgb_test, r2_xgb_test = safe_final_metrics(
            y_test_orig, pred_xgb_test, "XGBoost"
        )
        rmse_ens_test, mae_ens_test, r2_ens_test = safe_final_metrics(
            y_test_orig, pred_ens_test, "Ensemble"
        )

        print("\n=== Final Holdout Results ===")
        print(f"TabNet   | RMSE: {rmse_tab_test:.4f} | MAE: {mae_tab_test:.4f} | R²: {r2_tab_test:.4f}")
        print(f"XGBoost  | RMSE: {rmse_xgb_test:.4f} | MAE: {mae_xgb_test:.4f} | R²: {r2_xgb_test:.4f}")
        print(f"Ensemble | RMSE: {rmse_ens_test:.4f} | MAE: {mae_ens_test:.4f} | R²: {r2_ens_test:.4f}")

        # Save holdout metrics to CSV
        holdout_df = pd.DataFrame({
            "model": ["TabNet", "XGBoost", "Ensemble"],
            "rmse":  [rmse_tab_test, r2_xgb_test, r2_ens_test],
            "mae":   [mae_tab_test, mae_xgb_test, mae_ens_test],
            "r2":    [r2_tab_test, r2_xgb_test, r2_ens_test]
        })
        holdout_df.to_csv("holdout_metrics_fixed.csv", index=False)

        # Save test‐set predictions (on the original‐count scale)
        if not df_test_full.empty:
            results_df = pd.DataFrame({
                "ward_code":     df_test_full["ward_code_orig"].values,
                "year":          df_test_full["year_orig"].values,
                "month":         df_test_full["month_orig"].values,
                "actual":        y_test_orig,        # invert-log here
                "pred_tabnet":   pred_tab_test,
                "pred_xgboost":  pred_xgb_test,
                "pred_ensemble": pred_ens_test
            })
            results_df.to_csv("test_predictions_fixed.csv", index=False)
            print("Results saved to test_predictions_fixed.csv")

    except Exception as e:
        print(f"Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import warnings
from data_preparer_noscale import DataPreparerNoLeakage
from xgboost_model import XGBoostModel

# Import TabNet directly
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
    print("✅ TabNet imported successfully")
except ImportError as e:
    print(f"❌ TabNet import failed: {e}")
    TABNET_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet")


def validate_realistic_performance(y_true, y_pred, model_name=""):
    """Validate that performance is realistic for crime prediction"""
    r2 = r2_score(y_true, y_pred)
    
    if r2 < 0:
        print(f"🚨 CRITICAL: {model_name} R² = {r2:.4f} - MODEL IS BROKEN!")
        return False
    elif r2 > 0.6:
        print(f"🚨 WARNING: {model_name} R² = {r2:.4f} - LIKELY OVERFITTED")
        return False
    elif r2 > 0.35:
        print(f"✅ {model_name}: R² = {r2:.4f} - EXCELLENT and realistic")
        return True
    elif r2 > 0.25:
        print(f"✅ {model_name}: R² = {r2:.4f} - GOOD and realistic")
        return True
    elif r2 > 0.15:
        print(f"⚠️  {model_name}: R² = {r2:.4f} - Acceptable")
        return True
    else:
        print(f"❌ {model_name}: R² = {r2:.4f} - Needs improvement")
        return False


def debug_tabnet_inputs(X_train, y_train, X_val, y_val, stage=""):
    """Comprehensive TabNet input debugging"""
    print(f"\n=== TabNet Input Debug ({stage}) ===")
    
    # Training data
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"X_train NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"y_train NaN: {np.isnan(y_train).sum()}, Inf: {np.isinf(y_train).sum()}")
    
    # Validation data
    print(f"X_val shape: {X_val.shape}, dtype: {X_val.dtype}")
    print(f"y_val shape: {y_val.shape}, dtype: {y_val.dtype}")
    
    # Feature variance check
    feature_vars = np.var(X_train, axis=0)
    zero_var_features = (feature_vars < 1e-8).sum()
    print(f"Zero variance features: {zero_var_features}")
    
    return zero_var_features == 0


def fix_tabnet_data_aggressive(X_train, y_train, X_val, y_val):
    """Aggressively fix all TabNet data issues"""
    print("🔧 Aggressively fixing TabNet data...")
    
    # Convert to float32 (TabNet's preferred type)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    # Replace NaN/Inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e3, neginf=-1e3)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e6, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e3, neginf=-1e3)
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Clip extreme values
    X_train = np.clip(X_train, -1e3, 1e3)
    y_train = np.clip(y_train, 0, 1e3)  # Crime counts should be non-negative and reasonable
    X_val = np.clip(X_val, -1e3, 1e3)
    y_val = np.clip(y_val, 0, 1e3)
    
    # Remove zero variance features
    feature_vars = np.var(X_train, axis=0)
    var_mask = feature_vars > 1e-6  # More lenient threshold
    
    if var_mask.sum() < X_train.shape[1]:
        print(f"Removing {(~var_mask).sum()} zero-variance features")
        X_train = X_train[:, var_mask]
        X_val = X_val[:, var_mask]
    
    # Ensure minimum data size
    if X_train.shape[0] < 100:
        print("⚠️ Warning: Very small training set")
    
    return X_train, y_train, X_val, y_val, var_mask


def create_bulletproof_tabnet(n_features, device):
    """Create TabNet with MORE COMPLEX but stable parameters"""
    
    # COMPLEX but stable parameters
    tabnet_params = {
        'n_d': 32,                   # Larger network (was 8)
        'n_a': 32,                   # Larger network (was 8)
        'n_steps': 5,                # More steps for better feature selection (was 3)
        'gamma': 1.3,                # Standard feature reuse
        'cat_idxs': [],              # No categorical features for now
        'cat_dims': [],              # No categorical dimensions
        'cat_emb_dim': 1,            # Minimal embedding
        'n_independent': 3,          # More independent layers (was 2)
        'n_shared': 3,               # More shared layers (was 2)
        'epsilon': 1e-15,            # Numerical stability
        'momentum': 0.02,            # Low momentum
        'lambda_sparse': 5e-4,       # Less sparsity for more feature usage (was 1e-3)
        'seed': 42,                  # Reproducibility
        'verbose': 1,                # Show progress
        'device_name': device
    }
    
    # Enhanced training parameters
    fit_params = {
        'X_train': None,             # Will be set later
        'y_train': None,             # Will be set later
        'eval_set': None,            # Will be set later
        'eval_name': ['val'],        # Validation name
        'eval_metric': ['rmse'],     # Evaluation metric
        'max_epochs': 150,           # More epochs for complex model (was 50)
        'patience': 25,              # More patience for complex model (was 10)
        'batch_size': 512,           # Larger batches (was 256)
        'virtual_batch_size': 256,   # Larger virtual batch size (was 128)
        'num_workers': 0,            # No multiprocessing
        'drop_last': False,          # Keep all data
        'callbacks': None,           # No callbacks
        'pin_memory': False,         # No pinned memory
        'from_unsupervised': None,   # No unsupervised pretraining
        'warm_start': False,         # No warm start
        'augmentations': None,       # No augmentations
        'compute_importance': True   # Compute feature importance
    }
    
    return tabnet_params, fit_params


def comprehensive_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    errors = y_true - y_pred
    underpred_pct = (errors > 0).mean() * 100
    bias = (y_pred.mean() - y_true.mean()) / y_true.mean() * 100 if y_true.mean() != 0 else 0
    
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else 0
    
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2, 'correlation': correlation,
        'underpred_pct': underpred_pct, 'bias_pct': bias,
        'pred_mean': y_pred.mean(), 'actual_mean': y_true.mean(),
        'pred_std': y_pred.std(), 'actual_std': y_true.std()
    }


def encode_categorical_columns(X_df: pd.DataFrame):
    X_df_encoded = X_df.copy()
    label_encoders = {}
    
    for col in X_df_encoded.columns:
        if X_df_encoded[col].dtype in ["object", "category"]:
            le = LabelEncoder()
            X_df_encoded[col] = X_df_encoded[col].astype(str).fillna("unknown")
            X_df_encoded[col] = le.fit_transform(X_df_encoded[col])
            label_encoders[col] = le
    
    return X_df_encoded, label_encoders


def fix_data_issues(X, y):
    X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
    
    # Conservative cleaning
    X_df = X_df.fillna(X_df.median())
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(X_df.median())
    
    # Fix target
    y_fixed = np.nan_to_num(y, nan=np.median(y[~np.isnan(y)]) if not np.isnan(y).all() else 0)
    y_fixed = np.clip(y_fixed, 0, np.percentile(y_fixed, 99))
    
    return X_df.values, y_fixed


def main():
    csv_path = "ward_london.csv"
    target_col = "burglary_count"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("=== BULLETPROOF TABNET FIX ===")
    
    if not TABNET_AVAILABLE:
        print("❌ TabNet not available. Install with: pip install pytorch-tabnet")
        return
    
    try:
        preparer = DataPreparerNoLeakage(csv_path, target_col)
        df_train_full, df_test_full, train_end_date = preparer.preprocess_split_aware()
        
        print(f"Train period: up to {train_end_date}")
        print(f"Training samples: {len(df_train_full)}")
        print(f"Test samples: {len(df_test_full)}")

        X_train_df, y_train = preparer.prepare_features(df_train_full)
        X_test_df, y_test = preparer.prepare_features(df_test_full)
        
        if X_test_df.empty:
            print("WARNING: No test data available!")
            return

        print(f"Feature count: {X_train_df.shape[1]}")

        # Clean identifier columns
        to_drop = ["ward_code", "ward_code_orig", "year_orig", "month_orig", "actual"]
        X_train_clean = X_train_df.drop(columns=to_drop, errors="ignore")
        X_test_clean = X_test_df.drop(columns=to_drop, errors="ignore")

        # Categorical encoding
        cat_cols = [c for c in X_train_clean.columns if X_train_clean[c].dtype in ["object", "category"]]
        
        if cat_cols:
            X_train_encoded, label_encoders = encode_categorical_columns(X_train_clean)
            X_test_encoded = X_test_clean.copy()
            
            for col in cat_cols:
                if col in label_encoders:
                    le = label_encoders[col]
                    X_test_encoded[col] = X_test_encoded[col].astype(str).fillna("unknown")
                    mask = X_test_encoded[col].isin(le.classes_)
                    X_test_encoded.loc[mask, col] = le.transform(X_test_encoded.loc[mask, col])
                    X_test_encoded.loc[~mask, col] = -1
        else:
            X_train_encoded, X_test_encoded = X_train_clean.copy(), X_test_clean.copy()
            label_encoders = {}

        # Data cleaning
        X_train_values, y_train_clean = fix_data_issues(X_train_encoded.values, y_train)
        X_test_values, y_test_clean = fix_data_issues(X_test_encoded.values, y_test)

        # Use RobustScaler for TabNet (handles outliers better)
        scaler_tabnet = RobustScaler()
        scaler_xgb = StandardScaler()
        
        X_train_tabnet = scaler_tabnet.fit_transform(X_train_values)
        X_test_tabnet = scaler_tabnet.transform(X_test_values)
        
        X_train_xgb = scaler_xgb.fit_transform(X_train_values)
        X_test_xgb = scaler_xgb.transform(X_test_values)

        print(f"Final features: {X_train_tabnet.shape[1]}")

        # === TRAIN/VAL SPLIT ===
        split_point = int(len(X_train_tabnet) * 0.8)
        X_tr_tab, X_val_tab = X_train_tabnet[:split_point], X_train_tabnet[split_point:]
        X_tr_xgb, X_val_xgb = X_train_xgb[:split_point], X_train_xgb[split_point:]
        y_tr, y_val = y_train_clean[:split_point], y_train_clean[split_point:]

        # === FIX TABNET DATA ===
        print("\n🔧 Preparing TabNet data...")
        X_tr_tab_fixed, y_tr_fixed, X_val_tab_fixed, y_val_fixed, var_mask = fix_tabnet_data_aggressive(
            X_tr_tab, y_tr, X_val_tab, y_val
        )
        
        # *** CRITICAL FIX: RESHAPE TARGET TO 2D FOR TABNET ***
        y_tr_fixed = y_tr_fixed.reshape(-1, 1)      # Convert to 2D
        y_val_fixed = y_val_fixed.reshape(-1, 1)    # Convert to 2D
        
        print(f"Fixed target shapes: y_train={y_tr_fixed.shape}, y_val={y_val_fixed.shape}")
        
        # Apply same mask to test data
        X_test_tab_fixed = X_test_tabnet[:, var_mask].astype(np.float32)
        X_test_tab_fixed = np.nan_to_num(X_test_tab_fixed, nan=0.0, posinf=1e3, neginf=-1e3)
        X_test_tab_fixed = np.clip(X_test_tab_fixed, -1e3, 1e3)
        
        # Debug inputs
        if not debug_tabnet_inputs(X_tr_tab_fixed, y_tr_fixed, X_val_tab_fixed, y_val_fixed, "Final"):
            print("⚠️ Data quality issues detected but proceeding...")

        # === CREATE BULLETPROOF TABNET ===
        print("\n🛡️ Creating bulletproof TabNet...")
        tabnet_params, fit_params = create_bulletproof_tabnet(X_tr_tab_fixed.shape[1], device)
        
        final_predictions = {}

        # === TRAIN TABNET WITH MAXIMUM SAFETY ===
        print("\n=== TRAINING TABNET (BULLETPROOF) ===")
        
        tabnet_success = False
        try:
            # Create TabNet model
            tabnet_model = TabNetRegressor(**tabnet_params)
            
            print(f"TabNet created with {X_tr_tab_fixed.shape[1]} features")
            print(f"Training data shape: {X_tr_tab_fixed.shape}")
            print(f"Validation data shape: {X_val_tab_fixed.shape}")
            
            # Fit the model with COMPLEX parameters
            print("Starting TabNet training...")
            tabnet_model.fit(
                X_train=X_tr_tab_fixed,
                y_train=y_tr_fixed,
                eval_set=[(X_val_tab_fixed, y_val_fixed)],
                eval_name=['val'],
                eval_metric=['rmse'],
                max_epochs=150,              # More epochs
                patience=25,                 # More patience
                batch_size=512,              # Larger batches
                virtual_batch_size=256,      # Larger virtual batches
                num_workers=0,
                drop_last=False
            )
            
            print("✅ TabNet training completed!")
            
            # Validate on validation set
            val_pred_tab = tabnet_model.predict(X_val_tab_fixed)
            val_pred_tab = val_pred_tab.ravel()  # Convert back to 1D for metrics
            y_val_1d = y_val_fixed.ravel()       # Convert back to 1D for metrics
            
            val_metrics_tab = comprehensive_metrics(y_val_1d, val_pred_tab)
            print(f"TabNet Validation R²: {val_metrics_tab['r2']:.4f}")
            
            if val_metrics_tab['r2'] > -0.1:  # Not completely broken
                # Make test predictions
                pred_tab_test = tabnet_model.predict(X_test_tab_fixed)
                pred_tab_test = pred_tab_test.ravel()  # Convert back to 1D
                pred_tab_test = np.maximum(pred_tab_test, 0)  # Ensure non-negative
                
                final_predictions['tabnet'] = pred_tab_test
                tabnet_success = True
                print("✅ TabNet predictions successful!")
            else:
                print("❌ TabNet validation failed - R² too low")
                
        except Exception as e:
            print(f"❌ TabNet failed: {str(e)}")
            print("📋 Full error details:")
            import traceback
            traceback.print_exc()

        # === TRAIN XGBOOST (BACKUP) ===
        print("\n=== TRAINING XGBOOST ===")
        
        # Enhanced XGBoost to match TabNet complexity
        xgb_params = {
            'n_estimators': 800,         # More trees to match TabNet complexity
            'max_depth': 8,              # Deeper trees
            'learning_rate': 0.03,       # Lower LR for more trees
            'subsample': 0.85,           # Better sampling
            'colsample_bytree': 0.85,    # Better feature sampling
            'colsample_bylevel': 0.8,    # Additional sampling
            'reg_alpha': 0.3,            # Balanced regularization
            'reg_lambda': 0.7,           
            'random_state': 42,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'eval_metric': 'rmse'
        }
        
        try:
            xgb_final = XGBoostModel(device)
            xgb_final.train(X_tr_xgb, y_tr, xgb_params, eval_set=(X_val_xgb, y_val))
            
            val_pred_xgb = xgb_final.predict(X_val_xgb)
            val_metrics_xgb = comprehensive_metrics(y_val, val_pred_xgb)
            print(f"XGBoost Validation R²: {val_metrics_xgb['r2']:.4f}")
            
            pred_xgb_test = xgb_final.predict(X_test_xgb)
            final_predictions['xgboost'] = pred_xgb_test
            print("✅ XGBoost successful!")
            
        except Exception as e:
            print(f"❌ XGBoost failed: {e}")
            final_predictions['xgboost'] = np.full(len(X_test_xgb), y_train_clean.mean())

        # === ENSEMBLE WITH OPTIMIZED WEIGHTS ===
        if len(final_predictions) >= 2:
            if tabnet_success:
                # Both models working - optimize weights based on validation performance
                if 'tabnet' in final_predictions and 'xgboost' in final_predictions:
                    # Compare validation performance to set weights
                    if val_metrics_tab['r2'] > val_metrics_xgb['r2']:
                        # TabNet is better - give it more weight
                        pred_ensemble = 0.6 * final_predictions['tabnet'] + 0.4 * final_predictions['xgboost']
                        print("Using TabNet-favored ensemble (60-40)")
                    elif val_metrics_xgb['r2'] > val_metrics_tab['r2'] + 0.05:
                        # XGBoost is significantly better
                        pred_ensemble = 0.4 * final_predictions['tabnet'] + 0.6 * final_predictions['xgboost']
                        print("Using XGBoost-favored ensemble (40-60)")
                    else:
                        # Similar performance - balanced
                        pred_ensemble = 0.5 * final_predictions['tabnet'] + 0.5 * final_predictions['xgboost']
                        print("Using balanced ensemble (50-50)")
                else:
                    pred_ensemble = 0.5 * final_predictions['tabnet'] + 0.5 * final_predictions['xgboost']
                    print("Using balanced ensemble (50-50)")
            else:
                # Only XGBoost working
                pred_ensemble = final_predictions['xgboost']
                print("Using XGBoost only (TabNet failed)")
        else:
            pred_ensemble = list(final_predictions.values())[0]

        # === FINAL EVALUATION ===
        y_test_final = y_test_clean

        print(f"\n=== BULLETPROOF RESULTS (FINAL TEST SET) ===")
        
        best_r2 = -999
        
        # Individual model results
        for model_name, predictions in final_predictions.items():
            metrics = comprehensive_metrics(y_test_final, predictions)
            is_realistic = validate_realistic_performance(y_test_final, predictions, model_name.upper())
            
            print(f"{model_name.upper():>10} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
            print(f"{'':>10} | Corr: {metrics['correlation']:.4f} | Bias: {metrics['bias_pct']:.1f}%")
            
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']

        # Ensemble results
        if len(final_predictions) >= 2:
            metrics_ens = comprehensive_metrics(y_test_final, pred_ensemble)
            is_realistic_ens = validate_realistic_performance(y_test_final, pred_ensemble, "ENSEMBLE")
            
            print(f"{'ENSEMBLE':>10} | RMSE: {metrics_ens['rmse']:.4f} | MAE: {metrics_ens['mae']:.4f} | R²: {metrics_ens['r2']:.4f}")
            print(f"{'':>10} | Corr: {metrics_ens['correlation']:.4f} | Bias: {metrics_ens['bias_pct']:.1f}%")
            
            if metrics_ens['r2'] > best_r2:
                best_r2 = metrics_ens['r2']

        print(f"\n🏆 BEST R²: {best_r2:.4f}")
        
        if tabnet_success:
            print("🎉 TABNET IS WORKING!")
        else:
            print("⚠️ TabNet failed but XGBoost is working")
            print("\n🔍 TabNet Debugging Tips:")
            print("1. Check pytorch-tabnet version: pip install pytorch-tabnet==4.1.0")
            print("2. Try CPU device instead of CUDA")
            print("3. Reduce batch size further")
            print("4. Check for feature scaling issues")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
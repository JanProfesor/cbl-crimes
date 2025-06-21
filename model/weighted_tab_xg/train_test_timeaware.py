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
import matplotlib.pyplot as plt
import seaborn as sns
import os

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


def find_optimal_ensemble_weights(pred1, pred2, y_true, metric='rmse', n_points=101):
    """
    Find optimal ensemble weights by grid search optimization
    
    Parameters:
    pred1, pred2: predictions from model 1 and model 2
    y_true: actual values
    metric: optimization metric ('rmse', 'mae', or 'r2')
    n_points: number of weight combinations to test
    
    Returns:
    best_weight: optimal weight for model 1 (model 2 gets 1-best_weight)
    best_score: best score achieved
    all_weights: all tested weights
    all_scores: all scores for plotting
    """
    weights = np.linspace(0, 1, n_points)
    scores = []
    
    for w in weights:
        ensemble_pred = w * pred1 + (1 - w) * pred2
        
        if metric == 'rmse':
            score = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        elif metric == 'mae':
            score = mean_absolute_error(y_true, ensemble_pred)
        elif metric == 'r2':
            score = -r2_score(y_true, ensemble_pred)  # Negative because we want to minimize
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        scores.append(score)
    
    best_idx = np.argmin(scores)
    best_weight = weights[best_idx]
    best_score = scores[best_idx]
    
    return best_weight, best_score, weights, scores


def plot_ensemble_optimization(weights, scores, best_weight, best_score, metric='rmse', save_dir="feature_importance_plots"):
    """
    Plot the ensemble weight optimization curve
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(weights, scores, 'b-', linewidth=2, label=f'{metric.upper()} vs Weight')
    plt.axvline(best_weight, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal Weight = {best_weight:.3f}')
    plt.axhline(best_score, color='red', linestyle=':', alpha=0.7)
    
    plt.xlabel('TabNet Weight (XGBoost Weight = 1 - TabNet Weight)', fontsize=12)
    plt.ylabel(f'Validation {metric.upper()}', fontsize=12)
    plt.title(f'Ensemble Weight Optimization\nBest {metric.upper()}: {best_score:.4f} at TabNet Weight: {best_weight:.3f}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'ensemble_weight_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Ensemble optimization plot saved to {save_dir}/ensemble_weight_optimization.png")


def plot_ensemble_analysis(weights, rmse_scores, mae_scores, r2_scores, 
                          best_weight_rmse, best_weight_mae, best_weight_r2, 
                          save_dir="feature_importance_plots"):
    """
    Plot comprehensive ensemble analysis across all metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE plot
    axes[0].plot(weights, rmse_scores, 'b-', linewidth=2)
    axes[0].axvline(best_weight_rmse, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('TabNet Weight')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title(f'RMSE Optimization\nBest: {best_weight_rmse:.3f}')
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(weights, mae_scores, 'g-', linewidth=2)
    axes[1].axvline(best_weight_mae, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('TabNet Weight')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'MAE Optimization\nBest: {best_weight_mae:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    # R² plot (convert back to positive)
    r2_scores_pos = [-score for score in r2_scores]
    axes[2].plot(weights, r2_scores_pos, 'purple', linewidth=2)
    axes[2].axvline(best_weight_r2, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('TabNet Weight')
    axes[2].set_ylabel('R²')
    axes[2].set_title(f'R² Optimization\nBest: {best_weight_r2:.3f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive ensemble analysis saved to {save_dir}/ensemble_comprehensive_analysis.png")


def plot_feature_importances(feature_names, tabnet_importances=None, xgb_importances=None, 
                           ensemble_importances=None, tabnet_success=False, save_dir="results"):
    """
    Plot feature importances for TabNet, XGBoost, and ensemble models
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. TabNet Feature Importance Plot
    if tabnet_success and tabnet_importances is not None:
        # Get top 15 features for TabNet
        tabnet_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': tabnet_importances
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=tabnet_df, x='Importance', y='Feature', palette='viridis')
        plt.title('TabNet Feature Importance (Top 15)', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tabnet_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ TabNet feature importance plot saved to {save_dir}/tabnet_feature_importance.png")
    
    # 2. XGBoost Feature Importance Plot
    if xgb_importances is not None:
        # Get top 15 features for XGBoost
        xgb_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_importances
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=xgb_df, x='Importance', y='Feature', palette='plasma')
        plt.title('XGBoost Feature Importance (Top 15)', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'xgboost_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ XGBoost feature importance plot saved to {save_dir}/xgboost_feature_importance.png")
    
    # 3. Ensemble Feature Importance Plot
    if ensemble_importances is not None:
        # Get top 15 features for Ensemble
        ensemble_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': ensemble_importances
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=ensemble_df, x='Importance', y='Feature', palette='coolwarm')
        plt.title('Ensemble Feature Importance (Top 15)', fontsize=16, fontweight='bold')
        plt.xlabel('Weighted Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ensemble_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Ensemble feature importance plot saved to {save_dir}/ensemble_feature_importance.png")
    
    # 4. Comparison Plot (if both models are available)
    if tabnet_success and tabnet_importances is not None and xgb_importances is not None:
        # Create comparison of top 10 features from each model
        top_features = set(
            list(tabnet_df.head(10)['Feature']) + 
            list(xgb_df.head(10)['Feature'])
        )
        
        comparison_data = []
        for feature in top_features:
            tabnet_imp = tabnet_df[tabnet_df['Feature'] == feature]['Importance'].iloc[0] if feature in tabnet_df['Feature'].values else 0
            xgb_imp = xgb_df[xgb_df['Feature'] == feature]['Importance'].iloc[0] if feature in xgb_df['Feature'].values else 0
            comparison_data.append({
                'Feature': feature,
                'TabNet': tabnet_imp,
                'XGBoost': xgb_imp
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(['TabNet', 'XGBoost'], ascending=False).head(12)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparison_df['TabNet'], width, label='TabNet', alpha=0.8)
        bars2 = ax.bar(x + width/2, comparison_df['XGBoost'], width, label='XGBoost', alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Feature Importance', fontsize=12)
        ax.set_title('TabNet vs XGBoost Feature Importance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Model comparison plot saved to {save_dir}/model_comparison_feature_importance.png")


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
        
        # Apply same mask to XGBoost data to ensure feature consistency
        X_test_xgb_fixed = X_test_xgb[:, var_mask]
        X_tr_xgb = X_tr_xgb[:, var_mask]
        X_val_xgb = X_val_xgb[:, var_mask]
        X_test_xgb = X_test_xgb_fixed
        
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
        if len(final_predictions) >= 2 and tabnet_success:
            print("\n=== OPTIMIZING ENSEMBLE WEIGHTS ===")
            
            # Get validation predictions for weight optimization
            val_pred_tab = tabnet_model.predict(X_val_tab_fixed).ravel()
            val_pred_xgb = xgb_final.predict(X_val_xgb)
            y_val_1d = y_val_fixed.ravel()
            
            # Find optimal weights using validation data
            best_weight_rmse, best_rmse, weights, rmse_scores = find_optimal_ensemble_weights(
                val_pred_tab, val_pred_xgb, y_val_1d, metric='rmse', n_points=101
            )
            
            best_weight_mae, best_mae, _, mae_scores = find_optimal_ensemble_weights(
                val_pred_tab, val_pred_xgb, y_val_1d, metric='mae', n_points=101
            )
            
            best_weight_r2, best_r2, _, r2_scores = find_optimal_ensemble_weights(
                val_pred_tab, val_pred_xgb, y_val_1d, metric='r2', n_points=101
            )
            
            print(f"📊 Optimal weights found:")
            print(f"   RMSE optimization: TabNet={best_weight_rmse:.3f}, XGBoost={1-best_weight_rmse:.3f} (RMSE: {best_rmse:.4f})")
            print(f"   MAE optimization:  TabNet={best_weight_mae:.3f}, XGBoost={1-best_weight_mae:.3f} (MAE: {best_mae:.4f})")
            print(f"   R² optimization:   TabNet={best_weight_r2:.3f}, XGBoost={1-best_weight_r2:.3f} (R²: {-best_r2:.4f})")
            
            # Use RMSE-optimized weights (most common for ensemble optimization)
            optimal_weight = best_weight_rmse
            
            # Create optimized ensemble predictions
            pred_ensemble = optimal_weight * final_predictions['tabnet'] + (1 - optimal_weight) * final_predictions['xgboost']
            
            print(f"🎯 Using RMSE-optimized ensemble: TabNet={optimal_weight:.3f}, XGBoost={1-optimal_weight:.3f}")
            
            # Plot optimization curve
            plot_ensemble_optimization(weights, rmse_scores, best_weight_rmse, best_rmse, 'rmse')
            
            # Create comprehensive ensemble analysis plots
            plot_ensemble_analysis(weights, rmse_scores, mae_scores, r2_scores, 
                                  best_weight_rmse, best_weight_mae, best_weight_r2)
            
        elif len(final_predictions) >= 1:
            # Only one model available
            if 'xgboost' in final_predictions:
                pred_ensemble = final_predictions['xgboost']
                optimal_weight = 0.0
                print("🔄 Using XGBoost only (TabNet failed)")
            else:
                pred_ensemble = final_predictions['tabnet']
                optimal_weight = 1.0
                print("🔄 Using TabNet only (XGBoost failed)")
        else:
            print("❌ No models available for ensemble")
            return

        # === FEATURE IMPORTANCE ANALYSIS ===
        print("\n=== EXTRACTING FEATURE IMPORTANCES ===")
        
        # Get feature names (after cleaning and encoding)
        feature_names = list(X_train_clean.columns)
        
        # Apply the same variable mask that was used for TabNet to get consistent feature names
        if 'var_mask' in locals():
            feature_names_masked = [feature_names[i] for i in range(len(feature_names)) if var_mask[i]]
        else:
            feature_names_masked = feature_names
        
        # Extract TabNet feature importances
        tabnet_importances = None
        if tabnet_success:
            try:
                tabnet_importances = tabnet_model.feature_importances_
                print(f"✅ TabNet feature importances extracted: shape {tabnet_importances.shape}")
            except Exception as e:
                print(f"⚠️ Could not extract TabNet importances: {e}")
                tabnet_success = False
        
        # Extract XGBoost feature importances
        xgb_importances = None
        if 'xgb_final' in locals() and hasattr(xgb_final, 'feature_importances'):
            try:
                xgb_importances = xgb_final.feature_importances()
                print(f"✅ XGBoost feature importances extracted: shape {xgb_importances.shape}")
            except Exception as e:
                print(f"⚠️ Could not extract XGBoost importances: {e}")
        
        # Create ensemble feature importances (weighted average using optimal weights)
        ensemble_importances = None
        if tabnet_success and tabnet_importances is not None and xgb_importances is not None:
            # Use the mathematically optimal weights
            ensemble_importances = optimal_weight * tabnet_importances + (1 - optimal_weight) * xgb_importances
            print(f"✅ Created optimized ensemble importances ({optimal_weight:.3f}-{1-optimal_weight:.3f})")
        elif xgb_importances is not None:
            # Only XGBoost available
            ensemble_importances = xgb_importances
            optimal_weight = 0.0
            print("✅ Using XGBoost importances as ensemble (TabNet failed)")
        elif tabnet_importances is not None:
            # Only TabNet available
            ensemble_importances = tabnet_importances
            optimal_weight = 1.0
            print("✅ Using TabNet importances as ensemble (XGBoost failed)")
        
        # Generate feature importance plots
        plot_feature_importances(
            feature_names=feature_names_masked,
            tabnet_importances=tabnet_importances,
            xgb_importances=xgb_importances,
            ensemble_importances=ensemble_importances,
            tabnet_success=tabnet_success,
            save_dir="feature_importance_plots"
        )

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
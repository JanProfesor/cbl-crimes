# REQUIREMENTS:
# pip install pandas scikit-learn torch pytorch-tabnet optuna tqdm xgboost matplotlib seaborn

import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ―――――――――――――――――――――――――――
# 0. Load & preprocess
# ―――――――――――――――――――――――――――
df = pd.read_csv("processed/final_dataset_residential_burglary.csv")
df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
df = df.sort_values(['ward_code','date'])

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

TARGET = 'burglary_count'

# Feature engineering: add more lags and rolling statistics
# 1,3,6-month lags
for lag in (1, 2, 3, 6, 12):
    df[f'{TARGET}_lag_{lag}'] = df.groupby('ward_code')[TARGET].shift(lag)

# Rolling statistics (3, 6, 12 month windows)
for window in [3, 6, 12]:
    # Rolling mean/std
    df[f'{TARGET}_roll_mean_{window}'] = df.groupby('ward_code')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    df[f'{TARGET}_roll_std_{window}'] = df.groupby('ward_code')[TARGET].transform(
        lambda x: x.rolling(window, min_periods=1).std())

# Add month, year as features
df['month_num'] = df['date'].dt.month
df['year_num'] = df['date'].dt.year

# Cyclical month encoding
ang = 2*np.pi*df['month_num']/12
df['month_sin'] = np.sin(ang)
df['month_cos'] = np.cos(ang)

# drop missing
print(f"Rows before dropping NAs: {len(df)}")
df.dropna(inplace=True)
print(f"Rows after dropping NAs: {len(df)}")

# Feature selection - exclude raw date, year and month
features_to_drop = [TARGET, 'date', 'year', 'month']
X = df.drop(columns=features_to_drop)

# Pre-process
X['ward_code'] = X['ward_code'].astype('category').cat.codes
cat_idxs = [X.columns.get_loc('ward_code')]
cat_dims = [X['ward_code'].nunique()]

# Scale numerical features 
num_cols = [col for col in X.columns if col != 'ward_code']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# make y array
y = df[TARGET].values

# 70/30 hold-out for final ensemble
X_full, X_hold, y_full, y_hold = train_test_split(
    X.values, y, test_size=0.3, shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ―――――――――――――――――――――――――――
# 1. Quick 3-fold CV baseline
# ―――――――――――――――――――――――――――
base_cv = TimeSeriesSplit(n_splits=3)
print("\n=== Baseline 3-fold CV (fixed params) ===")
base_scores = {"fold": [], "rmse": [], "mae": [], "r2": []}

for fold,(tr,val) in enumerate(base_cv.split(X_full),1):
    Xt, Xv = X_full[tr], X_full[val]
    yt, yv = y_full[tr], y_full[val]
    
    # Reshape for TabNet
    yt_reshaped = yt.reshape(-1, 1)
    
    m = TabNetRegressor(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam, optimizer_params={'lr':1e-2},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size':50, 'gamma':0.9},
        mask_type='sparsemax', device_name=device
    )
    m.fit(
        Xt, yt_reshaped,
        eval_set=[(Xv, yv.reshape(-1, 1))], eval_name=['val'], eval_metric=['rmse'],
        max_epochs=30, patience=5,
        batch_size=4096, virtual_batch_size=4096
    )
    p = m.predict(Xv).reshape(-1)
    rmse = np.sqrt(mean_squared_error(yv, p))
    mae = mean_absolute_error(yv, p)
    r2 = r2_score(yv, p)
    
    base_scores["fold"].append(fold)
    base_scores["rmse"].append(rmse)
    base_scores["mae"].append(mae)
    base_scores["r2"].append(r2)
    
    print(f" Fold {fold}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

# ―――――――――――――――――――――――――――
# 2. Optuna tuning (3-fold inside)
# ―――――――――――――――――――――――――――
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(direction="minimize", pruner=pruner)

def objective(trial):
    # Expanded search space
    n_d   = trial.suggest_int('n_d', 8, 128)
    n_a   = trial.suggest_int('n_a', 8, 128)  # Separate n_a parameter
    n_st  = trial.suggest_int('n_steps', 3, 15)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    lr    = trial.suggest_float('lr', 1e-4, 5e-1, log=True)
    wd    = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    bs    = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192])
    vbs   = trial.suggest_categorical('virtual_batch_size', [128, 256, 512, 1024])
    mom   = trial.suggest_float('momentum', 0.01, 0.99)
    
    # Try different mask types
    mask_type = trial.suggest_categorical('mask_type', ['sparsemax', 'entmax'])
    
    # Add cat_emb_dim as a tunable parameter
    cat_emb_dim = trial.suggest_int('cat_emb_dim', 1, 5)
    
    rmses, maes, r2s = [], [], []
    inner_cv = TimeSeriesSplit(n_splits=3)
    
    for tr,val in inner_cv.split(X_full):
        Xt, Xv = X_full[tr], X_full[val]
        yt, yv = y_full[tr], y_full[val]
        
        # Reshape for TabNet
        yt_reshaped = yt.reshape(-1, 1)
        
        m = TabNetRegressor(
            n_d=n_d, n_a=n_a, n_steps=n_st, gamma=gamma,
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr':lr, 'weight_decay':wd},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size':10, 'gamma':0.8},
            momentum=mom, mask_type=mask_type, device_name=device
        )
        m.fit(
            Xt, yt_reshaped,
            eval_set=[(Xv, yv.reshape(-1, 1))], eval_name=['val'], eval_metric=['rmse'],
            max_epochs=30, patience=5,
            batch_size=bs, virtual_batch_size=vbs
        )
        p = m.predict(Xv).reshape(-1)
        rmse = np.sqrt(mean_squared_error(yv, p))
        rmses.append(rmse)
        maes.append(mean_absolute_error(yv, p))
        r2s.append(r2_score(yv, p))

    # log extra metrics
    trial.set_user_attr('avg_mae', float(np.mean(maes)))
    trial.set_user_attr('avg_r2', float(np.mean(r2s)))
    
    return float(np.mean(rmses))

print("\n=== Running Optuna ===")
study.optimize(objective, n_trials=30, timeout=3600)  # Increased to 30 trials with 1hr timeout

# dump Optuna results
tr_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs', 'state'))
tr_df.to_csv("optuna_trials.csv", index=False)

with open("optuna_best.json", "w") as f:
    json.dump({
        "best_rmse": study.best_value,
        "best_params": study.best_trial.params,
        "best_mae": study.best_trial.user_attrs['avg_mae'],
        "best_r2": study.best_trial.user_attrs['avg_r2']
    }, f, indent=2)

print(f"\nBest trial #{study.best_trial.number}:")
print(f" RMSE = {study.best_value:.4f}")
print(f" MAE  = {study.best_trial.user_attrs['avg_mae']:.4f}")
print(f" R²   = {study.best_trial.user_attrs['avg_r2']:.4f}")
print("Parameters:", study.best_trial.params)

# Visualize Optuna results
plt.figure(figsize=(12, 8))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.savefig('optuna_history.png')

plt.figure(figsize=(16, 12))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.tight_layout()
plt.savefig('optuna_param_importances.png')

# ―――――――――――――――――――――――――――
# 3. Find optimal ensemble weights
# ―――――――――――――――――――――――――――
print("\n=== Finding optimal ensemble weights ===")

# 3.1 Train TabNet with best params on full data
bp = study.best_trial.params
tabnet = TabNetRegressor(
    n_d=bp['n_d'], n_a=bp['n_a'], n_steps=bp['n_steps'], gamma=bp['gamma'],
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=bp['cat_emb_dim'],
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr':bp['lr'], 'weight_decay':bp['weight_decay']},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={'step_size':10, 'gamma':0.8},
    momentum=bp['momentum'], mask_type=bp['mask_type'], device_name=device
)

# Reshape y_full for tabnet
y_full_reshaped = y_full.reshape(-1, 1)

tabnet.fit(
    X_full, y_full_reshaped,
    max_epochs=50, patience=10,
    batch_size=bp['batch_size'], virtual_batch_size=bp['virtual_batch_size']
)
pred_tab = tabnet.predict(X_hold).reshape(-1)

# Get feature importances from TabNet
feat_importances = tabnet.feature_importances_
feat_names = list(X.columns)
feature_importance_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': feat_importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('TabNet Feature Importances')
plt.tight_layout()
plt.savefig('tabnet_feature_importances.png')

# 3.2 Train XGBoost on same full training
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist' if device=='cuda' else 'hist',
    random_state=42
)
xgb_model.fit(X_full, y_full)
pred_xgb = xgb_model.predict(X_hold)

# Plot XGBoost feature importances
xgb_importances = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': xgb_importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df.head(15))
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.savefig('xgb_feature_importances.png')

# 3.3 Find optimal blend weights
def blend_score(w, p1, p2, y_true):
    """Calculate RMSE of blended predictions with weight w for p1"""
    blend = w * p1 + (1-w) * p2
    return np.sqrt(mean_squared_error(y_true, blend))

# Search for optimal weight for TabNet
weights = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00
rmses = [blend_score(w, pred_tab, pred_xgb, y_hold) for w in weights]
best_w = weights[np.argmin(rmses)]
best_rmse = min(rmses)

plt.figure(figsize=(10, 6))
plt.plot(weights, rmses)
plt.axvline(x=best_w, color='r', linestyle='--')
plt.xlabel('TabNet Weight')
plt.ylabel('RMSE')
plt.title(f'Ensemble Weight Optimization (Best TabNet Weight: {best_w:.2f})')
plt.grid(True)
plt.savefig('ensemble_weights.png')

# 3.4 Blend with optimal weights
pred_ens = best_w * pred_tab + (1 - best_w) * pred_xgb

# 3.5 Evaluate
rmse_tab = np.sqrt(mean_squared_error(y_hold, pred_tab))
mae_tab = mean_absolute_error(y_hold, pred_tab)
r2_tab = r2_score(y_hold, pred_tab)

rmse_xgb = np.sqrt(mean_squared_error(y_hold, pred_xgb))
mae_xgb = mean_absolute_error(y_hold, pred_xgb)
r2_xgb = r2_score(y_hold, pred_xgb)

rmse_ens = np.sqrt(mean_squared_error(y_hold, pred_ens))
mae_ens = mean_absolute_error(y_hold, pred_ens)
r2_ens = r2_score(y_hold, pred_ens)

print(f"TabNet RMSE = {rmse_tab:.3f}, MAE = {mae_tab:.3f}, R² = {r2_tab:.3f}")
print(f"XGBoost RMSE = {rmse_xgb:.3f}, MAE = {mae_xgb:.3f}, R² = {r2_xgb:.3f}")
print(f"Ensemble RMSE = {rmse_ens:.3f}, MAE = {mae_ens:.3f}, R² = {r2_ens:.3f}")
print(f"Optimal TabNet weight: {best_w:.3f}")

# save ensemble metrics
with open("ensemble_metrics.json", "w") as f:
    json.dump({
        "tabnet": {"rmse": rmse_tab, "mae": mae_tab, "r2": r2_tab},
        "xgboost": {"rmse": rmse_xgb, "mae": mae_xgb, "r2": r2_xgb},
        "ensemble": {"rmse": rmse_ens, "mae": mae_ens, "r2": r2_ens},
        "ensemble_weight_tabnet": float(best_w)
    }, f, indent=2)

# ―――――――――――――――――――――――――――
# 4. Plot Predictions vs Actual
# ―――――――――――――――――――――――――――
# Create a dataframe for plotting
plot_df = pd.DataFrame({
    'Actual': y_hold,
    'TabNet': pred_tab,
    'XGBoost': pred_xgb,
    'Ensemble': pred_ens
})

# Take a sample of points to avoid overcrowding
sample_size = min(500, len(plot_df))
plot_sample = plot_df.sample(sample_size, random_state=42)

# Plot 1: Actual vs Predicted scatter plot
plt.figure(figsize=(12, 10))
plt.scatter(plot_sample['Actual'], plot_sample['Ensemble'], alpha=0.5, label='Ensemble')
plt.scatter(plot_sample['Actual'], plot_sample['TabNet'], alpha=0.5, label='TabNet')
plt.scatter(plot_sample['Actual'], plot_sample['XGBoost'], alpha=0.5, label='XGBoost')

# Perfect prediction line
max_val = max(plot_sample['Actual'].max(), plot_sample['Ensemble'].max()) * 1.1
min_val = min(plot_sample['Actual'].min(), plot_sample['Ensemble'].min()) * 0.9
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Burglary Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')

# Plot 2: Error distribution
plt.figure(figsize=(15, 8))
errors_tab = plot_df['TabNet'] - plot_df['Actual']
errors_xgb = plot_df['XGBoost'] - plot_df['Actual']
errors_ens = plot_df['Ensemble'] - plot_df['Actual']

sns.kdeplot(errors_tab, label=f'TabNet (RMSE={rmse_tab:.3f})')
sns.kdeplot(errors_xgb, label=f'XGBoost (RMSE={rmse_xgb:.3f})')
sns.kdeplot(errors_ens, label=f'Ensemble (RMSE={rmse_ens:.3f})')
plt.axvline(x=0, color='k', linestyle='--')
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.title('Error Distribution of Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('error_distribution.png')

print("✅ Analysis complete. Saved metrics and visualizations.")
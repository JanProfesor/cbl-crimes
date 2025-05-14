# REQUIREMENTS:
# pip install pandas scikit-learn torch pytorch-tabnet optuna tqdm

import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import optuna
from optuna.pruners import HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback

# ―――――――――――――――――――――――――――――――
# 0. Load & preprocess data
# ―――――――――――――――――――――――――――――――
df = pd.read_csv("processed/final_dataset_residential_burglary.csv")
df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
df = df.sort_values(['ward_code','date'])

TARGET = 'burglary_count'
# add 1-, 3-, and 6-month lags
for lag in (1, 3, 6):
    df[f'{TARGET}_lag_{lag}'] = df.groupby('ward_code')[TARGET].shift(lag)

# cyclical month features
df['month_num'] = df['date'].dt.month
ang = 2 * np.pi * df['month_num'] / 12
df['month_sin'] = np.sin(ang)
df['month_cos'] = np.cos(ang)

# drop rows missing lags
df.dropna(subset=[f'{TARGET}_lag_{l}' for l in (1,3,6)], inplace=True)

# features & target
X = df.drop(columns=[TARGET, 'date'])
y = df[TARGET].values.reshape(-1, 1)

# encode ward_code
X['ward_code'] = X['ward_code'].astype('category').cat.codes
cat_idxs = [X.columns.get_loc('ward_code')]
cat_dims = [X['ward_code'].nunique()]

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ―――――――――――――――――――――――――――――――
# 1. Quick 3-fold CV baseline
# ―――――――――――――――――――――――――――――――
tscv = TimeSeriesSplit(n_splits=3)
cv_metrics = {'rmse': [], 'mae': [], 'r2': []}

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X.values[tr_idx], X.values[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = TabNetRegressor(
        n_d=16, n_a=16, n_steps=5, gamma=1.5,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam, optimizer_params={'lr':1e-2},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size':50,'gamma':0.9},
        mask_type='sparsemax',
        device_name=device
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)], eval_name=['val'], eval_metric=['rmse'],
        max_epochs=30, patience=5,
        batch_size=512, virtual_batch_size=128
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)

    cv_metrics['rmse'].append(rmse)
    cv_metrics['mae'].append(mae)
    cv_metrics['r2'].append(r2)
    print(f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

print("\nCV mean ± std:")
for m, vals in cv_metrics.items():
    arr = np.array(vals)
    print(f" {m.upper()}: {arr.mean():.2f} ± {arr.std():.2f}")

# save CV metrics
cv_df = pd.DataFrame(cv_metrics)
cv_df.to_csv('cv_metrics.csv', index=False)
print("✅ Saved cross-validation metrics to cv_metrics.csv")

# ―――――――――――――――――――――――――――――――
# 2. Optuna with 3-fold CV & pruning
# ―――――――――――――――――――――――――――――――
pruner = HyperbandPruner(min_resource=5, max_resource=20, reduction_factor=3)
study = optuna.create_study(direction="minimize", pruner=pruner)

def objective(trial):
    # hyperparameter space
    n_d        = trial.suggest_int('n_d', 8, 64)
    n_steps    = trial.suggest_int('n_steps', 3, 10)
    gamma      = trial.suggest_float('gamma', 1.0, 2.5)
    lr         = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])

    rmses = []
    inner_cv = TimeSeriesSplit(n_splits=3)
    for tr_idx, val_idx in inner_cv.split(X):
        X_tr, X_val = X.values[tr_idx], X.values[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = TabNetRegressor(
            n_d=n_d, n_a=n_d, n_steps=n_steps, gamma=gamma,
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': lr, 'weight_decay': weight_decay},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size':10, 'gamma':0.8},
            mask_type='sparsemax',
            device_name=device
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)], eval_name=['val'], eval_metric=['rmse'],
            max_epochs=20, patience=3,
            batch_size=batch_size, virtual_batch_size=128
        )

        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    return float(np.mean(rmses))

# run optimization
study.optimize(objective, n_trials=10)

# report & save
print("\nBest RMSE:", study.best_value)
print("Best params:", study.best_trial.params)

trials_df = study.trials_dataframe()
trials_df.to_csv("optuna_trials.csv", index=False)
with open("optuna_best.json", "w") as f:
    json.dump({
        "best_rmse": study.best_value,
        "best_params": study.best_trial.params
    }, f, indent=2)
print("✅ Saved optuna_trials.csv and optuna_best.json")

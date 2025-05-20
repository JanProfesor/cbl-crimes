# REQUIREMENTS:
# pip install pandas scikit-learn torch pytorch-tabnet optuna tqdm xgboost

import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import xgboost as xgb
import optuna
from optuna.pruners import HyperbandPruner

# ―――――――――――――――――――――――――――
# 0. Load & preprocess
# ―――――――――――――――――――――――――――
df = pd.read_csv("processed/final_dataset_residential_burglary.csv")
df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
df = df.sort_values(['ward_code','date'])

TARGET = 'burglary_count'
# 1,3,6-month lags
for lag in (1,3,6):
    df[f'{TARGET}_lag_{lag}'] = df.groupby('ward_code')[TARGET].shift(lag)
# cyclical month
df['month_num'] = df['date'].dt.month
ang = 2*np.pi*df['month_num']/12
df['month_sin'] = np.sin(ang)
df['month_cos'] = np.cos(ang)
# drop missing
df.dropna(subset=[f'{TARGET}_lag_{l}' for l in (1,3,6)], inplace=True)


X = df.drop(columns=[TARGET,'date'])
# make y two-dimensional for TabNet
y = df[TARGET].values.reshape(-1, 1)

# encode
X['ward_code'] = X['ward_code'].astype('category').cat.codes
cat_idxs = [X.columns.get_loc('ward_code')]
cat_dims = [X['ward_code'].nunique()]

# 70/30 hold-out for final ensemble
# both y_full and y_hold will now already be shape (n,1)
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
for fold,(tr,val) in enumerate(base_cv.split(X_full),1):
    Xt, Xv = X_full[tr], X_full[val]
    yt, yv = y_full[tr], y_full[val]
    m = TabNetRegressor(
        n_d=32,n_a=32,n_steps=5,gamma=1.5,
        cat_idxs=cat_idxs,cat_dims=cat_dims,cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,optimizer_params={'lr':1e-2},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size':50,'gamma':0.9},
        mask_type='sparsemax',device_name=device
    )
    m.fit(
        Xt, yt,
        eval_set=[(Xv,yv)],eval_name=['val'],eval_metric=['rmse'],
        max_epochs=30,patience=5,
        batch_size=4096,virtual_batch_size=4096
    )
    p = m.predict(Xv)
    print(f" Fold {fold}: RMSE={np.sqrt(mean_squared_error(yv,p)):.3f}, "
          f"MAE={mean_absolute_error(yv,p):.3f}, R²={r2_score(yv,p):.3f}")

# ―――――――――――――――――――――――――――
# 2. Optuna tuning (3-fold inside)
# ―――――――――――――――――――――――――――
pruner = HyperbandPruner(min_resource=5, max_resource=20, reduction_factor=3)
study = optuna.create_study(direction="minimize", pruner=pruner)

def objective(trial):
    # search space
    n_d   = trial.suggest_int('n_d',16,64)
    n_st  = trial.suggest_int('n_steps',3,10)
    gamma = trial.suggest_float('gamma',1.0,2.5)
    lr    = trial.suggest_float('lr',1e-4,1e-1,log=True)
    wd    = trial.suggest_float('weight_decay',1e-6,1e-2,log=True)
    bs    = trial.suggest_categorical('batch_size',[1024,2048,4096])

    rmses, maes, r2s = [], [], []
    inner_cv = TimeSeriesSplit(n_splits=3)
    for tr,val in inner_cv.split(X_full):
        Xt, Xv = X_full[tr], X_full[val]
        yt, yv = y_full[tr], y_full[val]
        m = TabNetRegressor(
            n_d=n_d,n_a=n_d,n_steps=n_st,gamma=gamma,
            cat_idxs=cat_idxs,cat_dims=cat_dims,cat_emb_dim=1,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr':lr,'weight_decay':wd},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size':10,'gamma':0.8},
            mask_type='sparsemax',device_name=device
        )
        m.fit(
            Xt, yt,
            eval_set=[(Xv,yv)],eval_name=['val'],eval_metric=['rmse'],
            max_epochs=30,patience=5,
            batch_size=bs,virtual_batch_size=1024
        )
        p = m.predict(Xv)
        rmse = np.sqrt(mean_squared_error(yv,p))
        rmses.append(rmse)
        maes.append(mean_absolute_error(yv,p))
        r2s.append(r2_score(yv,p))

    # log extra metrics
    trial.set_user_attr('avg_mae', float(np.mean(maes)))
    trial.set_user_attr('avg_r2', float(np.mean(r2s)))

    return float(np.mean(rmses))

print("\n=== Running Optuna ===")
study.optimize(objective, n_trials=10)

# dump Optuna results
tr_df = study.trials_dataframe(attrs=('number','value','params','user_attrs'))
tr_df.to_csv("optuna_trials.csv", index=False)
with open("optuna_best.json","w") as f:
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

# ―――――――――――――――――――――――――――
# 3. Final ensemble on hold-out
# ―――――――――――――――――――――――――――
print("\n=== Ensemble on 30% hold-out ===")
# 3.1 Train TabNet with best params on full data
bp = study.best_trial.params
tabnet = TabNetRegressor(
    n_d=bp['n_d'], n_a=bp['n_d'], n_steps=bp['n_steps'], gamma=bp['gamma'],
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr':bp['lr'],'weight_decay':bp['weight_decay']},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={'step_size':10,'gamma':0.8},
    mask_type='sparsemax',device_name=device
)
tabnet.fit(
    X_full, y_full,
    max_epochs=30,patience=5,
    batch_size=bp['batch_size'],virtual_batch_size=1024
)
pred_tab = tabnet.predict(X_hold).reshape(-1)

# 3.2 Train XGBoost on same full training
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist' if device=='cuda' else 'hist'
)
xgb_model.fit(X_full, y_full)
pred_xgb = xgb_model.predict(X_hold)

# 3.3 Blend by simple average
pred_ens = 0.5*pred_tab + 0.5*pred_xgb

# 3.4 Evaluate
rmse_e = np.sqrt(mean_squared_error(y_hold, pred_ens))
mae_e  = mean_absolute_error(y_hold, pred_ens)
r2_e   = r2_score(y_hold, pred_ens)
print(f"Ensemble RMSE = {rmse_e:.3f}")
print(f"Ensemble MAE  = {mae_e:.3f}")
print(f"Ensemble R²   = {r2_e:.3f}")

# save ensemble metrics
with open("ensemble_metrics.json","w") as f:
    json.dump({"rmse":rmse_e,"mae":mae_e,"r2":r2_e}, f, indent=2)
print("✅ Saved optuna_trials.csv, optuna_best.json and ensemble_metrics.json")

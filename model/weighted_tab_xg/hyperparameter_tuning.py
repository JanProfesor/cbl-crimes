import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from optuna.pruners import HyperbandPruner
from tabnet_model import TabNetModel
from xgboost_model import XGBoostModel

def tune_tabnet(X, y, cat_idxs, cat_dims, device, n_trials=30):
    pruner = HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)
    study = optuna.create_study(direction='minimize', pruner=pruner)

    def objective(trial):
        params = {
            'n_d': trial.suggest_int('n_d', 8, 128),
            'n_a': trial.suggest_int('n_a', 8, 128),
            'n_steps': trial.suggest_int('n_steps', 3, 15),
            'gamma': trial.suggest_float('gamma', 1.0, 3.0),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 0, 1e-2),
            'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
            'n_shared': trial.suggest_int('n_shared', 1, 3),
            'n_independent': trial.suggest_int('n_independent', 1, 3),
            'cat_idxs': cat_idxs,
            'cat_dims': cat_dims,
            'cat_emb_dim': trial.suggest_int('cat_emb_dim', 1, 10)
        }
        cv = TimeSeriesSplit(n_splits=3)
        fold_mses = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
            Xtr, Xv = X[tr_idx], X[val_idx]
            ytr, yv = y[tr_idx], y[val_idx]
            model = TabNetModel(cat_idxs, cat_dims, device)
            model.train(Xtr, ytr, params)
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)
            fold_mses.append(mse)
            trial.report(mse, step=fold + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_mses))

    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_xgboost(X, y, device, n_trials=30):
    pruner = HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)
    study = optuna.create_study(direction='minimize', pruner=pruner)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'lr': trial.suggest_float('lr', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
        cv = TimeSeriesSplit(n_splits=3)
        fold_mses = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
            Xtr, Xv = X[tr_idx], X[val_idx]
            ytr, yv = y[tr_idx], y[val_idx]
            model = XGBoostModel(device)
            model.train(Xtr, ytr, params)
            preds = model.predict(Xv)
            mse = mean_squared_error(yv, preds)
            fold_mses.append(mse)
            trial.report(mse, step=fold + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_mses))

    study.optimize(objective, n_trials=n_trials)
    return study.best_params

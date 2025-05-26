
import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tabnet_model import TabNetModel
from xgboost_model import XGBoostModel

def tune_tabnet(X, y, cat_idxs, cat_dims, device, n_trials=30):
    def objective(trial):
        # define TabNet search space
        params = {
            'n_d': trial.suggest_int('n_d', 8, 128),
            'n_a': trial.suggest_int('n_a', 8, 128),
            'n_steps': trial.suggest_int('n_steps', 3, 15),
            'gamma': trial.suggest_float('gamma', 1.0, 3.0),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192]),
            'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [128, 256, 512, 1024]),
            'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
            'cat_emb_dim': trial.suggest_int('cat_emb_dim', 1, 5),
            'momentum': trial.suggest_float('momentum', 0.01, 0.99),
        }
        cv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for tr_idx, val_idx in cv.split(X):
            Xtr, Xv = X[tr_idx], X[val_idx]
            ytr, yv = y[tr_idx], y[val_idx]
            model = TabNetModel(cat_idxs, cat_dims, device)
            model.train(Xtr, ytr, params, X_val=Xv, y_val=yv)
            preds = model.predict(Xv)
            rmses.append(mean_squared_error(yv, preds))
        return float(np.mean(rmses))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_xgboost(X, y, device, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'tree_method': 'gpu_hist' if device=='cuda' else 'hist',
            'random_state': 42
        }
        cv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for tr_idx, val_idx in cv.split(X):
            Xtr, Xv = X[tr_idx], X[val_idx]
            ytr, yv = y[tr_idx], y[val_idx]
            model = XGBoostModel(device)
            model.train(Xtr, ytr, params)
            preds = model.predict(Xv)
            rmses.append(mean_squared_error(yv, preds))
        return float(np.mean(rmses))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

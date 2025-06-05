import torch
from pytorch_tabnet.tab_model import TabNetRegressor

class TabNetModel:
    def __init__(self, cat_idxs, cat_dims, device):
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.device = device
        self.model = None

    def train(self, X_train, y_train, params, X_val=None, y_val=None):
        self.model = TabNetRegressor(
            n_d=params['n_d'], n_a=params['n_a'], n_steps=params['n_steps'], gamma=params['gamma'],
            cat_idxs=self.cat_idxs, cat_dims=self.cat_dims, cat_emb_dim=params.get('cat_emb_dim', 1),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr':params['lr'], 'weight_decay':params.get('weight_decay', 1e-5)},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={'step_size':10, 'gamma':0.8},
            momentum=params.get('momentum', 0.3), mask_type=params.get('mask_type', 'sparsemax'),
            device_name=self.device
        )
        eval_set = [(X_val, y_val.reshape(-1, 1))] if X_val is not None else None
        self.model.fit(
            X_train, y_train.reshape(-1, 1),
            eval_set=eval_set, eval_name=['val'] if eval_set else None, eval_metric=['rmse'],
            max_epochs=30, patience=5,
            batch_size=params.get('batch_size', 4096), virtual_batch_size=params.get('virtual_batch_size', 4096)
        )

    def predict(self, X):
        return self.model.predict(X).reshape(-1)

    def feature_importances(self):
        return self.model.feature_importances_
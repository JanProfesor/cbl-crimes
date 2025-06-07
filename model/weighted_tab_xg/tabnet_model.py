import torch
from pytorch_tabnet.tab_model import TabNetRegressor

class TabNetModel:
    def __init__(self, cat_idxs, cat_dims, device):
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.device = device
        self.model = None

    def train(self, X_train, y_train, params, X_val=None, y_val=None):
        """
        If X_val, y_val are provided, use them as eval_set so early stopping works.
        """
        self.model = TabNetRegressor(
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=params.get('cat_emb_dim', 1),

            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': params['lr'], 'weight_decay': params.get('weight_decay', 1e-5)},

            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={'mode':'min', 'factor':0.5, 'patience':10, 'verbose':False},

            momentum=params.get('momentum', 0.3),
            mask_type=params.get('mask_type', 'entmax'),   # switch to entmax
            lambda_sparse=params.get('lambda_sparse', 1e-3),
            device_name=self.device
        )

        eval_set = None
        eval_names = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val.reshape(-1,1))]
            eval_names = ['val']

        self.model.fit(
            X_train,
            y_train.reshape(-1, 1),
            eval_set=eval_set,
            eval_name=eval_names,
            eval_metric=['rmse'],
            max_epochs=params['max_epochs'],
            patience=params['patience'],
            batch_size=params['batch_size'],
            virtual_batch_size=params['virtual_batch_size']
        )


    def predict(self, X):
        return self.model.predict(X).reshape(-1)

    def feature_importances(self):
        return self.model.feature_importances_
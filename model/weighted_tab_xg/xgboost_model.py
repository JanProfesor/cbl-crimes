import xgboost as xgb

class XGBoostModel:
    def __init__(self, device):
        self.device = device
        self.model = None

    
    def train(self, X_train, y_train, params: dict, eval_set: tuple = None):
        """
        If eval_set is provided as (X_val, y_val), we’ll do early stopping.
        Otherwise, fit on the full X_train/y_train.
        """
        self.model = xgb.XGBRegressor(**params)
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_
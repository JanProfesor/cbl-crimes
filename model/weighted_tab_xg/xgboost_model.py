import xgboost as xgb

class XGBoostModel:
    def __init__(self, device):
        self.device = device
        self.model = None

    def train(self, X_train, y_train, params: dict, eval_set: tuple = None):
        """
        If eval_set is provided as (X_val, y_val), we'll use it for evaluation,
        but we remove unsupported kwargs (early_stopping_rounds) to avoid errors.
        """
        # Copy so we don’t mutate the caller’s dict
        params = params.copy()

        # Ensure accepted eval_metric and verbosity at construction
        params.setdefault('eval_metric', 'rmse')
        params.setdefault('verbosity', 0)
        # GPU flags (optional)
        params.setdefault('tree_method', 'gpu_hist')
        params.setdefault('predictor', 'gpu_predictor')

        # Instantiate the model
        self.model = xgb.XGBRegressor(**params)

        if eval_set is not None:
            X_val, y_val = eval_set
            # Remove unsupported early_stopping_rounds
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)]
                # early_stopping_rounds removed
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_

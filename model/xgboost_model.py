import xgboost as xgb

class XGBoostModel:
    def __init__(self, device):
        self.device = device
        self.model = None

    def train(self, X_train, y_train):
        self.model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='gpu_hist' if self.device=='cuda' else 'hist',
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_
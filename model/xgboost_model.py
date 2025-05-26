import xgboost as xgb

class XGBoostModel:
    def __init__(self, device):
        self.device = device
        self.model = None

    
    def train(self, X_train, y_train, params: dict):
    
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

uk_data = pd.read_csv('data_file_uk')
london_data = pd.read_csv('data_file_london')

# the features to predict the target
features = ['tmax', 'tmin', 'af', 'rain', 'sun', 'crime',
            'education', 'employment', 'environment', 'health',
            'housing', 'income', 'burglary_count_lag1', 'house_price']
target = 'burglary_count'

# splitting data for UK (without London)
X_uk = uk_data[features]
y_uk = np.log1p(uk_data[target])  # log-transform

# London
from sklearn.model_selection import train_test_split
X_london = london_data[features]
y_london = np.log1p(london_data[target])  # log-transform

# train/test split for evaluation
X_london_train, X_london_test, y_london_train, y_london_test = train_test_split(
    X_london, y_london, test_size=0.3, random_state=42
)


# define class
class XGBoostCrimeCountModel:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'reg:squarederror',  # tells XGBoost we're doing regression
            'eval_metric': 'rmse',            # Root Mean Squared Error
            'eta': 0.1,                       # learning rate
            'max_depth': 6,                   # depth of trees
            'subsample': 0.8                  # use 80% of rows per tree to reduce overfitting
        }
        self.uk_model = None
        self.final_model = None

    def train_uk(self, X_uk, y_uk, num_boost_round=100):
        # Convert to DMatrix format (optimized for XGBoost)
        dtrain_uk = xgb.DMatrix(X_uk, label=y_uk)
        # Train UK model
        self.uk_model = xgb.train(self.params, dtrain_uk, num_boost_round=num_boost_round)

    def fine_tune_london(self, X_london, y_london, num_boost_round=50):
        # Check that UK model is trained first
        if self.uk_model is None:
            raise ValueError("UK model not trained. Call train_uk() first.")
        dtrain_london = xgb.DMatrix(X_london, label=y_london)
        # Fine-tune (continue training) from UK model
        self.final_model = xgb.train(
            self.params,
            dtrain_london,
            num_boost_round=num_boost_round,
            xgb_model=self.uk_model  # this is where transfer learning happens
        )

    def predict(self, X):
        if self.final_model is None:
            raise ValueError("Final model not trained. Call fine_tune_london() first.")
        dmatrix = xgb.DMatrix(X)
        return self.final_model.predict(dmatrix)

    def feature_importances(self):
        if self.final_model is None:
            raise ValueError("Model not trained.")
        return self.final_model.get_score(importance_type='gain')

# train on UK
model = XGBoostCrimeCountModel()
model.train_uk(X_uk, y_uk, num_boost_round=150)

# fine-tune on London
model.fine_tune_london(X_london_train, y_london_train, num_boost_round=50)

# predictions
y_pred_log = model.predict(X_london_test)    # still in log scale
y_pred = np.expm1(y_pred_log)                # convert back to normal scale

# performance metrics
y_true = np.expm1(y_london_test)
def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"R²  : {r2:.4f}")
    return rmse, mae, mse, r2
evaluate_predictions(y_true, y_pred)

# feature importance
importances = model.feature_importances()
sorted(importances.items(), key=lambda x: x[1], reverse=True)
for feature, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.2f}")

# save and load
model.final_model.save_model("XGBoost_model.json")
loaded_model=xgb.Booster()
loaded_model.load_model("XGBoost_model.json")
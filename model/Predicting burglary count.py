import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# preprocessing right skewed data
y_uk = np.log1p(y_uk)        # log(1 + x) handles 0s well
y_london = np.log1p(y_london)

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

    def forecast_future(self, last_known_df, steps=12, target_col='Burglary Count'):
        """
        Forecast future burglary counts using autoregression.
        - last_known_df: DataFrame with latest known features & target (must have lag columns).
        - steps: number of future months to forecast
        - target_col: original target column name (for inverse transform)
        """
        future_preds = []
        current_df = last_known_df.copy()

        for _ in range(steps):
            dmatrix = xgb.DMatrix(current_df.drop(columns=[target_col]))
            pred_log = self.final_model.predict(dmatrix)[0]
            pred = np.expm1(pred_log)
            future_preds.append(pred)

            # update lags: shift lag_1 to lag_2, etc.
            for i in reversed(range(2, 13)):
                current_df[f'lag_{i}'] = current_df[f'lag_{i - 1}']
            current_df['lag_1'] = pred_log  # keep in log scale for next input

            # update rolling features
            if 'rolling_3' in current_df.columns:
                last_lags = [current_df[f'lag_{i}'].values[0] for i in range(1, 4)]
                current_df['rolling_3'] = sum(last_lags) / 3
            if 'rolling_6' in current_df.columns:
                last_lags = [current_df[f'lag_{i}'].values[0] for i in range(1, 7)]
                current_df['rolling_6'] = sum(last_lags) / 6

            # advance date features
            current_month = current_df['month'].values[0]
            current_year = current_df['year'].values[0]
            next_month = current_month + 1
            next_year = current_year
            if next_month > 12:
                next_month = 1
                next_year += 1

            current_df['month'] = next_month
            current_df['year'] = next_year
            current_df['quarter'] = (next_month - 1) // 3 + 1

        return future_preds

# train on UK
model = XGBoostCrimeCountModel()
model.train_uk(X_uk, y_uk, num_boost_round=150)

# fine-tune on London
model.fine_tune_london(X_london, y_london, num_boost_round=50)

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
loaded_model = xgb.Booster()
loaded_model.load_model("XGBoost_model.json")

# future forecasting
target_col = 'Burglary Count'
last_known = X_london_test.copy()
last_known[target_col] = np.log1p(y_london_test.values)  # dummy target col
last_known_df = last_known.iloc[[-1]]  # last known row

future_forecasts = model.forecast_future(last_known_df, steps=12)

# Display
print("\nFuture Monthly Burglary Predictions:")
for i, val in enumerate(future_forecasts, 1):
    print(f"Month +{i}: {val:.2f}")

# Save to CSV
future_df = pd.DataFrame({
    "Month Ahead": list(range(1, 13)),
    "Predicted Burglary Count": future_forecasts
})
future_df.to_csv("future_burglary_forecast.csv", index=False)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(range(1, 13), future_forecasts, marker='o', label="Forecast")
plt.title("Burglary Count Forecast for Next 12 Months")
plt.xlabel("Months Ahead")
plt.ylabel("Predicted Burglary Count")
plt.grid(True)
plt.legend()
plt.show()

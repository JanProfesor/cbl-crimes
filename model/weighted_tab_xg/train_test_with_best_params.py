import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from data_preparer import DataPreparer
from tabnet_model import TabNetModel
from xgboost_model import XGBoostModel

def _build_identifier_df(csv_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.sort_values(["ward_code", "date"])
    for lag in (1, 2, 3, 6, 12):
        df[f"{target}_lag_{lag}"] = df.groupby("ward_code")[target].shift(lag)
    for window in (3, 6, 12):
        rollout = df.groupby("ward_code")[target].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        rollstd = df.groupby("ward_code")[target].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f"{target}_roll_mean_{window}"] = rollout
        df[f"{target}_roll_std_{window}"] = rollstd
    df["month_num"] = df["date"].dt.month
    df["year_num"] = df["date"].dt.year
    ang = 2 * np.pi * df["month_num"] / 12
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    df = df.dropna().reset_index(drop=True)
    return df[["ward_code", "year", "month"]].copy()

def encode_categorical_columns(X_df):
    """Encode any string/categorical columns to integers"""
    X_df_encoded = X_df.copy()
    label_encoders = {}
    
    for col in X_df_encoded.columns:
        if X_df_encoded[col].dtype == 'object' or X_df_encoded[col].dtype.name == 'category':
            print(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            X_df_encoded[col] = le.fit_transform(X_df_encoded[col].astype(str))
            label_encoders[col] = le
    
    return X_df_encoded, label_encoders

def main():
    csv_path = "processed/final_dataset_residential_burglary_reordered.csv"
    target_col = "burglary_count"
    ident_df = _build_identifier_df(csv_path, target_col)

    preparer = DataPreparer(csv_path, target_col)
    X_df, y, cat_idxs, cat_dims, _ = preparer.preprocess()
    
    
    X_df_encoded, label_encoders = encode_categorical_columns(X_df)
    X_values = X_df_encoded.values
    
    
    print("Data types after encoding:")
    print(X_df_encoded.dtypes)
    print("Any non-numeric values remaining:", X_df_encoded.select_dtypes(include=['object']).columns.tolist())
    
    n_total = X_values.shape[0]

    split_index = int(n_total * 0.7)
    X_train = X_values[:split_index]
    y_train = y[:split_index]
    X_hold = X_values[split_index:]
    y_hold = y[split_index:]
    ids_hold = ident_df.iloc[split_index:].reset_index(drop=True)

    with open("runs/best_run_yet/best_tabnet_params.json") as f:
        tabnet_params = json.load(f)
    with open("runs/best_run_yet/best_xgb_params.json") as f:
        xgb_params = json.load(f)

    ensemble_weight = 0.48
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    tabnet_rmse_list, xgb_rmse_list, ens_rmse_list = [], [], []
    tabnet_mae_list, xgb_mae_list, ens_mae_list = [], [], []
    tabnet_r2_list, xgb_r2_list, ens_r2_list = [], [], []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        tabnet = TabNetModel(cat_idxs, cat_dims, device)
        tabnet.train(X_tr, y_tr, tabnet_params)
        pred_tab_val = tabnet.predict(X_val)

        xgb_model = XGBoostModel(device)
        xgb_model.train(X_tr, y_tr, xgb_params)
        pred_xgb_val = xgb_model.predict(X_val)

        pred_ens_val = ensemble_weight * pred_tab_val + (1 - ensemble_weight) * pred_xgb_val

        rmse_tab = np.sqrt(mean_squared_error(y_val, pred_tab_val))
        rmse_xgb = np.sqrt(mean_squared_error(y_val, pred_xgb_val))
        rmse_ens = np.sqrt(mean_squared_error(y_val, pred_ens_val))

        mae_tab = mean_absolute_error(y_val, pred_tab_val)
        mae_xgb = mean_absolute_error(y_val, pred_xgb_val)
        mae_ens = mean_absolute_error(y_val, pred_ens_val)

        r2_tab = r2_score(y_val, pred_tab_val)
        r2_xgb = r2_score(y_val, pred_xgb_val)
        r2_ens = r2_score(y_val, pred_ens_val)

        tabnet_rmse_list.append(rmse_tab)
        xgb_rmse_list.append(rmse_xgb)
        ens_rmse_list.append(rmse_ens)

        tabnet_mae_list.append(mae_tab)
        xgb_mae_list.append(mae_xgb)
        ens_mae_list.append(mae_ens)

        tabnet_r2_list.append(r2_tab)
        xgb_r2_list.append(r2_xgb)
        ens_r2_list.append(r2_ens)

    print("TabNet 10-fold CV RMSE:", np.mean(tabnet_rmse_list))
    print("XGBoost 10-fold CV RMSE:", np.mean(xgb_rmse_list))
    print("Ensemble 10-fold CV RMSE:", np.mean(ens_rmse_list))

    tabnet_final = TabNetModel(cat_idxs, cat_dims, device)
    tabnet_final.train(X_train, y_train, tabnet_params)
    pred_tab_hold = tabnet_final.predict(X_hold)

    xgb_final = XGBoostModel(device)
    xgb_final.train(X_train, y_train, xgb_params)
    pred_xgb_hold = xgb_final.predict(X_hold)

    pred_ens_hold = ensemble_weight * pred_tab_hold + (1 - ensemble_weight) * pred_xgb_hold

    rmse_tab_hold = np.sqrt(mean_squared_error(y_hold, pred_tab_hold))
    rmse_xgb_hold = np.sqrt(mean_squared_error(y_hold, pred_xgb_hold))
    rmse_ens_hold = np.sqrt(mean_squared_error(y_hold, pred_ens_hold))

    mae_tab_hold = mean_absolute_error(y_hold, pred_tab_hold)
    mae_xgb_hold = mean_absolute_error(y_hold, pred_xgb_hold)
    mae_ens_hold = mean_absolute_error(y_hold, pred_ens_hold)

    r2_tab_hold = r2_score(y_hold, pred_tab_hold)
    r2_xgb_hold = r2_score(y_hold, pred_xgb_hold)
    r2_ens_hold = r2_score(y_hold, pred_ens_hold)

    print("Hold-out TabNet RMSE:", rmse_tab_hold)
    print("Hold-out XGBoost RMSE:", rmse_xgb_hold)
    print("Hold-out Ensemble RMSE:", rmse_ens_hold)

    results_df = pd.DataFrame({
        "ward": ids_hold["ward_code"],
        "year": ids_hold["year"],
        "month": ids_hold["month"],
        "actual": y_hold,
        "pred_tabnet": pred_tab_hold,
        "pred_xgboost": pred_xgb_hold,
        "pred_ensemble": pred_ens_hold
    })

    results_csv_path = "test_predictions_final.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Hold-out predictions saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
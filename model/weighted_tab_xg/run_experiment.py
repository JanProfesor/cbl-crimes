import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preparer import DataPreparer
from tabnet_model import TabNetModel
from xgboost_model import XGBoostModel
from ensemble_blender import EnsembleBlender
from utils import create_run_folder
from hyperparameter_tuning import tune_tabnet, tune_xgboost
import random


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

def main():
    run_dir = create_run_folder()
    csv_path = "processed/final_dataset_residential_burglary.csv"
    target_col = "burglary_count"
    ident_df = _build_identifier_df(csv_path, target_col)
    preparer = DataPreparer(csv_path, target_col)
    X_df, y, cat_idxs, cat_dims = preparer.preprocess()
    feature_cols = list(X_df.columns)
    X_values = X_df[feature_cols].values
    n_total = X_values.shape[0]
    split_index = int(n_total * 0.7)
    X_train = X_values[:split_index]
    y_train = y[:split_index]
    X_hold = X_values[split_index:]
    y_hold = y[split_index:]
    ids_hold = ident_df.iloc[split_index:].reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_tabnet_params = tune_tabnet(X_train, y_train, cat_idxs, cat_dims, device)
    with open(os.path.join(run_dir, "best_tabnet_params.json"), "w") as f:
        json.dump(best_tabnet_params, f, indent=2)
    best_xgb_params = tune_xgboost(X_train, y_train, device)
    with open(os.path.join(run_dir, "best_xgb_params.json"), "w") as f:
        json.dump(best_xgb_params, f, indent=2)
    tabnet = TabNetModel(cat_idxs, cat_dims, device)
    tabnet.train(X_train, y_train, best_tabnet_params)
    pred_tab = tabnet.predict(X_hold)
    try:
        feat_imp_tab = tabnet.feature_importances()
        df_imp_tab = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": feat_imp_tab
        }).sort_values("Importance", ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=df_imp_tab)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "tabnet_feature_importances.png"))
        plt.close()
    except AttributeError:
        pass
    xgb_model = XGBoostModel(device)
    xgb_model.train(X_train, y_train, best_xgb_params)
    pred_xgb = xgb_model.predict(X_hold)
    try:
        feat_imp_xgb = xgb_model.feature_importances()
        df_imp_xgb = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": feat_imp_xgb
        }).sort_values("Importance", ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=df_imp_xgb)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "xgb_feature_importances.png"))
        plt.close()
    except AttributeError:
        pass
    best_w, best_rmse = EnsembleBlender.find_best_weight(pred_tab, pred_xgb, y_hold)
    pred_ens = best_w * pred_tab + (1 - best_w) * pred_xgb
    all_weights = np.linspace(0, 1, 101)
    all_rmses = [
        np.sqrt(mean_squared_error(y_hold, w * pred_tab + (1 - w) * pred_xgb))
        for w in all_weights
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(all_weights, all_rmses)
    plt.axvline(best_w, color="r", linestyle="--")
    plt.xlabel("TabNet Weight")
    plt.ylabel("Hold‐out RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "ensemble_weight_optimization.png"))
    plt.close()
    rmse_tab = np.sqrt(mean_squared_error(y_hold, pred_tab))
    mae_tab = mean_absolute_error(y_hold, pred_tab)
    r2_tab = r2_score(y_hold, pred_tab)
    rmse_xgb = np.sqrt(mean_squared_error(y_hold, pred_xgb))
    mae_xgb = mean_absolute_error(y_hold, pred_xgb)
    r2_xgb = r2_score(y_hold, pred_xgb)
    rmse_ens = np.sqrt(mean_squared_error(y_hold, pred_ens))
    mae_ens = mean_absolute_error(y_hold, pred_ens)
    r2_ens = r2_score(y_hold, pred_ens)
    metrics_dict = {
        "tabnet":   {"rmse": float(rmse_tab), "mae": float(mae_tab), "r2": float(r2_tab)},
        "xgboost":  {"rmse": float(rmse_xgb), "mae": float(mae_xgb), "r2": float(r2_xgb)},
        "ensemble": {"rmse": float(rmse_ens), "mae": float(mae_ens), "r2": float(r2_ens)},
        "ensemble_weight_tabnet": float(best_w)
    }
    with open(os.path.join(run_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
    df_plot = pd.DataFrame({
        "Actual":   y_hold,
        "TabNet":   pred_tab,
        "XGBoost":  pred_xgb,
        "Ensemble": pred_ens
    })
    plt.figure(figsize=(10, 8))
    plt.hexbin(
        df_plot["Actual"],
        df_plot["Ensemble"],
        gridsize=50,
        cmap="Blues",
        mincnt=1,
    )
    plt.colorbar(label="Count")
    mx = max(df_plot["Actual"].max(), df_plot["Ensemble"].max()) * 1.05
    mn = min(df_plot["Actual"].min(), df_plot["Ensemble"].min()) * 0.95
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual Burglary Count")
    plt.ylabel("Ensemble Predicted Count")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "actual_vs_predicted_hexbin.png"))
    plt.close()
    errs_tab = pred_tab - y_hold
    errs_xgb = pred_xgb - y_hold
    errs_ens = pred_ens - y_hold
    plt.figure(figsize=(15, 8))
    sns.kdeplot(errs_tab, label=f"TabNet (RMSE={rmse_tab:.3f})")
    sns.kdeplot(errs_xgb, label=f"XGBoost (RMSE={rmse_xgb:.3f})")
    sns.kdeplot(errs_ens, label=f"Ensemble (RMSE={rmse_ens:.3f})")
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Prediction Error (Pred − Actual)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "error_distribution.png"))
    plt.close()
    results_df = pd.DataFrame({
        "ward":         ids_hold["ward_code"],
        "year":         ids_hold["year"],
        "month":        ids_hold["month"],
        "actual":       y_hold,
        "pred_tabnet":  pred_tab,
        "pred_xgboost": pred_xgb,
        "pred_ensemble": pred_ens
    })
    results_csv_path = os.path.join(run_dir, "test_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    ward_list = results_df["ward"].unique().tolist()
    random_ward = random.choice(ward_list)
    ward_df = results_df[results_df["ward"] == random_ward].copy()["ward"].unique().tolist()
    random_ward = random.choice(ward_list)["ward"].unique()
    ward_df = results_df[results_df["ward"] == random_ward].copy()
    ward_df["date"] = pd.to_datetime(dict(year=ward_df["year"], month=ward_df["month"], day=1))
    ward_df = ward_df.sort_values("date")
    plt.figure(figsize=(12, 6))
    plt.plot(ward_df["date"], ward_df["actual"], label="Actual", marker="o")
    plt.plot(ward_df["date"], ward_df["pred_ensemble"], label="Predicted", marker="x")
    plt.xlabel("Date")
    plt.ylabel("Burglary Count")
    plt.title(f"Ward {random_ward}: Actual vs Predicted Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"ward_{random_ward}_time_series.png"))
    plt.close()

if __name__ == "__main__":
    main()

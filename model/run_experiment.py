import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from data_preparer import DataPreparer
from tabnet_model import TabNetModel
from xgboost_model import XGBoostModel
from ensemble_blender import EnsembleBlender
from utils import create_run_folder

import torch

def main():
    # --- Setup ---
    run_dir = create_run_folder()
    print(f"Saving results to: {run_dir}")

    # --- Data ---
    data = DataPreparer("processed/final_dataset_residential_burglary.csv", "burglary_count")
    X, y, cat_idxs, cat_dims = data.preprocess()
    X_full, X_hold, y_full, y_hold = train_test_split(X.values, y, test_size=0.3, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- TabNet ---
    tabnet_params = {
        'n_d': 32, 'n_a': 32, 'n_steps': 5, 'gamma': 1.5,
        'lr': 1e-2, 'batch_size': 4096, 'virtual_batch_size': 4096,
        'mask_type': 'sparsemax', 'cat_emb_dim': 1
    }
    tabnet = TabNetModel(cat_idxs, cat_dims, device)
    tabnet.train(X_full, y_full, tabnet_params)
    pred_tab = tabnet.predict(X_hold)

    # TabNet feature importances
    feat_importances = tabnet.feature_importances()
    feat_names = list(X.columns)
    feature_importance_df = (
        sns.barplot(
            x='Importance', y='Feature',
            data=(lambda df: df.sort_values('Importance', ascending=False).head(15))(
                pd.DataFrame({'Feature': feat_names, 'Importance': feat_importances})
            )
        )
    )
    plt.title('TabNet Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'tabnet_feature_importances.png'))
    plt.close()

    # --- XGBoost ---
    xgb = XGBoostModel(device)
    xgb.train(X_full, y_full)
    pred_xgb = xgb.predict(X_hold)

    # XGBoost feature importances
    xgb_importances = xgb.feature_importances()
    xgb_importance_df = (
        sns.barplot(
            x='Importance', y='Feature',
            data=(lambda df: df.sort_values('Importance', ascending=False).head(15))(
                pd.DataFrame({'Feature': feat_names, 'Importance': xgb_importances})
            )
        )
    )
    plt.title('XGBoost Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'xgb_feature_importances.png'))
    plt.close()

    # --- Ensemble ---
    best_w, best_rmse = EnsembleBlender.find_best_weight(pred_tab, pred_xgb, y_hold)
    pred_ens = best_w * pred_tab + (1 - best_w) * pred_xgb

    # Plot ensemble weight search
    weights = np.linspace(0, 1, 101)
    rmses = [np.sqrt(mean_squared_error(y_hold, w*pred_tab + (1-w)*pred_xgb)) for w in weights]
    plt.figure(figsize=(10, 6))
    plt.plot(weights, rmses)
    plt.axvline(x=best_w, color='r', linestyle='--')
    plt.xlabel('TabNet Weight')
    plt.ylabel('RMSE')
    plt.title(f'Ensemble Weight Optimization (Best TabNet Weight: {best_w:.2f})')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'ensemble_weights.png'))
    plt.close()

    # --- Metrics ---
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
        "tabnet": {"rmse": rmse_tab, "mae": mae_tab, "r2": r2_tab},
        "xgboost": {"rmse": rmse_xgb, "mae": mae_xgb, "r2": r2_xgb},
        "ensemble": {"rmse": rmse_ens, "mae": mae_ens, "r2": r2_ens},
        "ensemble_weight_tabnet": float(best_w)
    }
    with open(os.path.join(run_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # --- Plot Predictions vs Actual ---
    
    plot_df = pd.DataFrame({
    'Actual': y_hold,
    'TabNet': pred_tab,
    'XGBoost': pred_xgb,
    'Ensemble': pred_ens
})

    plt.figure(figsize=(10, 8))
    plt.hexbin(plot_df['Actual'], plot_df['Ensemble'], gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(label='Count')
    max_val = max(plot_df['Actual'].max(), plot_df['Ensemble'].max()) * 1.05
    min_val = min(plot_df['Actual'].min(), plot_df['Ensemble'].min()) * 0.95
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual')
    plt.ylabel('Predicted (Ensemble)')
    plt.title('Actual vs Predicted Burglary Count (Ensemble)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'actual_vs_predicted_hexbin.png'))
    plt.close()

    # --- Error Distribution ---
    plt.figure(figsize=(15, 8))
    errors_tab = plot_df['TabNet'] - plot_df['Actual']
    errors_xgb = plot_df['XGBoost'] - plot_df['Actual']
    errors_ens = plot_df['Ensemble'] - plot_df['Actual']
    sns.kdeplot(errors_tab, label=f'TabNet (RMSE={rmse_tab:.3f})')
    sns.kdeplot(errors_xgb, label=f'XGBoost (RMSE={rmse_xgb:.3f})')
    sns.kdeplot(errors_ens, label=f'Ensemble (RMSE={rmse_ens:.3f})')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution of Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'error_distribution.png'))
    plt.close()

    print("✅ Analysis complete. Saved metrics and visualizations.")

if __name__ == "__main__":
    main()
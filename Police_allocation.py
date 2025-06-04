# Re-import necessary libraries after code execution state reset
import pandas as pd
import numpy as np
import json
from scipy.optimize import minimize

csv_path = "processed/final_dataset_residential_burglary.csv"
df = pd.read_csv(csv_path)

if 'ward_code' not in df.columns:
    df['ward_code'] = df['lsoa_code']  # fallback to another column as ward_code

df_grouped = df.groupby('ward_code', as_index=False)['burglary_count'].sum()

wards = df_grouped['ward_code']
burglary_predictions = df_grouped['burglary_count'].values
burglary_predictions[burglary_predictions == 0] = 1e-6  # prevent math issues

max_hours_per_ward = 800
total_budget = 30000

def objective(x, b):
    return -np.sum(b * np.log(1 + x))

constraints = [{'type': 'ineq', 'fun': lambda x: total_budget - np.sum(x)}]
bounds = [(0, max_hours_per_ward)] * len(burglary_predictions)
x0 = np.ones(len(burglary_predictions)) * (total_budget / len(burglary_predictions))

result = minimize(objective, x0, args=(burglary_predictions,), bounds=bounds, constraints=constraints)
df_grouped['allocated_patrol_hours'] = result.x

output_path = "processed/allocated_patrol_hours_output.csv"
df_grouped.to_csv(output_path, index=False)

output_path

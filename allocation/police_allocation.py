import pandas as pd
import numpy as np
from scipy.optimize import minimize

pred_path = "processed/test_predictions_fixed.csv"
df = pd.read_csv(pred_path)

if 'ward_code' not in df.columns:
    raise ValueError("expected 'ward_code' column in the prediction file")
if 'pred_ensemble' not in df.columns:
    raise ValueError("expected 'pred_ensemble' column in the prediction file")

df_grouped = df.groupby('ward_code', as_index=False)['pred_ensemble'].sum()

burglary_predictions = df_grouped['pred_ensemble'].values
burglary_predictions[burglary_predictions == 0] = 1e-6

num_wards = len(burglary_predictions)
total_budget = 30000  # total patrol hours to allocate

def objective(x, b):
    return -np.sum(b * np.log(1 + x))

bounds = [(0, None)] * num_wards

x0 = np.full(num_wards, total_budget / num_wards)

constraints = [{'type': 'ineq', 'fun': lambda x: total_budget - np.sum(x)}]

# run optimization
result = minimize(objective, x0, args=(burglary_predictions,), bounds=bounds, constraints=constraints)

if not result.success:
    raise RuntimeError("optimization failed to converge")

# store the result
df_grouped['allocated_patrol_hours'] = result.x

# save the output
output_path = "allocation/patrol_allocation_by_ward.csv"
df_grouped.to_csv(output_path, index=False)

print(f"saved ward-level patrol allocations to: {output_path}")

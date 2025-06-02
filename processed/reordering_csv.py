import pandas as pd

# 1. Load the original CSV
input_path = 'processed/final_dataset_residential_burglary.csv'
df = pd.read_csv(input_path)

# 2. Sort so that for each (year, month), all ward_code rows are grouped together
df_sorted = df.sort_values(['year', 'month', 'ward_code']).reset_index(drop=True)

# 3. (Optional) Verify the ordering—e.g., print the first few rows
print(df_sorted.head(20))

# 4. Save the reordered CSV
output_path = 'processed/final_dataset_residential_burglary_reordered.csv'
df_sorted.to_csv(output_path, index=False)

print(f"Reordered CSV saved to: {output_path}")

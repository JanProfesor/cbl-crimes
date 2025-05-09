import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")

file_path = 'processed/final_dataset.csv'
df = pd.read_csv(file_path)

df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

df = df[(df['date'].dt.year >= 2010) & (df['date'].dt.year <= 2022)]

unit_corrections = {
    'TX': 10, 'TN': 10, 'TG': 10,
    'SS': 10,
    'RR': 10
}
for col, factor in unit_corrections.items():
    if col in df.columns:
        df[col] = df[col] / factor

selected_cols = [
    'avg_max_temperature', 'max_temperature', 'min_max_temperature',
    'avg_min_temperature', 'avg_temperature',
    'total_rainfall', 'max_daily_rainfall', 'rainfall_std'
]
selected_cols = [col for col in selected_cols if col in df.columns]
df_main = df[selected_cols]

df_main = df_main.rename(columns={
    'avg_max_temperature': 'Avg Max Temp (°C)',
    'max_temperature': 'Max Temp (°C)',
    'min_max_temperature': 'Min Max Temp (°C)',
    'avg_min_temperature': 'Avg Min Temp (°C)',
    'avg_temperature': 'Mean Temp (°C)',
    'total_rainfall': 'Total Rainfall (mm)',
    'max_daily_rainfall': 'Max Daily Rainfall (mm)',
    'rainfall_std': 'Rainfall Std Dev'
})

quartiles = df_main.quantile([0.25, 0.5, 0.75])
q1, q2, q3 = quartiles.loc[0.25], quartiles.loc[0.5], quartiles.loc[0.75]
iqr = q3 - q1

outlier_counts = {}
for col in df_main.columns:
    lower = q1[col] - 1.5 * iqr[col]
    upper = q3[col] + 1.5 * iqr[col]
    outlier_counts[col] = df_main[(df_main[col] < lower) | (df_main[col] > upper)].shape[0]

summary = pd.DataFrame({
    'Q1': q1,
    'Median (Q2)': q2,
    'Q3': q3,
    'IQR': iqr,
    'Outlier Count': pd.Series(outlier_counts)
})
print(summary)

for col in df_main.columns:
    plt.figure(figsize=(8, 6))

    if 'Rainfall' in col and 'Total' in col:
        nonzero_data = df_main[df_main[col] > 0][[col]]
        nonzero_data.boxplot()
        plt.yscale('log')
        plt.title(f"{col} (log scale)", fontsize=14, fontweight='bold')
    else:
        df_main[[col]].boxplot()
        plt.title(col, fontsize=14, fontweight='bold')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

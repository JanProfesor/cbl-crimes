# 2. Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 3. Load dataset
file_path = '/content/drive/My Drive/CBL/London_Weather_2010-2022.csv'
df = pd.read_csv(file_path)

# 4. Filter dates
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
df = df[(df['DATE'].dt.year >= 2010) & (df['DATE'].dt.year <= 2022)]

# 5. Keep valid rows (all Q_* == 0)
quality_cols = [col for col in df.columns if col.startswith('Q_')]
df = df[df[quality_cols].eq(0).all(axis=1)].copy()

# 6. Convert units
unit_corrections = {
    'TX': 10, 'TN': 10, 'TG': 10,   # °C
    'SS': 10,                       # hours
    'RR': 10                        # mm
}
for col, factor in unit_corrections.items():
    if col in df.columns:
        df[col] = df[col] / factor

# 7. Select only key columns
selected_cols = ['TX', 'TN', 'TG', 'SS', 'RR', 'HU']
df_main = df[selected_cols]

# 8. Rename for readability
df_main = df_main.rename(columns={
    'TX': 'Max Temp (°C)',
    'TN': 'Min Temp (°C)',
    'TG': 'Mean Temp (°C)',
    'SS': 'Sunshine Duration (h)',
    'RR': 'Precipitation (mm)',
    'HU': 'Relative Humidity (%)'
})

# 9. Summary stats
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
summary


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
axes = axes.flatten()

for i, col in enumerate(df_main.columns):
    if col == "Precipitation (mm)":
        # Use log scale for y-axis; avoid log(0) by filtering
        nonzero_data = df_main[df_main[col] > 0][[col]]
        nonzero_data.boxplot(ax=axes[i])
        axes[i].set_yscale('log')
        axes[i].set_title(f"{col} (log scale)")
    else:
        df_main[[col]].boxplot(ax=axes[i])
        axes[i].set_title(col)
    
    axes[i].grid(True)

plt.suptitle("Box Plots by Variable: London Weather Data (2010–2022)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

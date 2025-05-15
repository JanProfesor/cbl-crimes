# Re-import necessary libraries after kernel reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Reload the datasets
all_crimes_df = pd.read_csv("processed/final_dataset_all.csv")
res_burglary_df = pd.read_csv("processed/final_dataset_residential_burglary.csv")

# Create datetime column
all_crimes_df['date'] = pd.to_datetime(all_crimes_df[['year', 'month']].assign(day=1))
res_burglary_df['date'] = pd.to_datetime(res_burglary_df[['year', 'month']].assign(day=1))

# Aggregate monthly crime counts
monthly_all = all_crimes_df.groupby('date')['burglary_count'].sum().reset_index(name='total_crimes')
monthly_res = res_burglary_df.groupby('date')['burglary_count'].sum().reset_index(name='residential_burglaries')

# Merge datasets
monthly_comparison = pd.merge(monthly_all, monthly_res, on='date')

# Apply Min-Max scaling
scaler = MinMaxScaler()
scaled_counts = scaler.fit_transform(monthly_comparison[['total_crimes', 'residential_burglaries']])
monthly_comparison['total_crimes_scaled'] = scaled_counts[:, 0]
monthly_comparison['res_burglaries_scaled'] = scaled_counts[:, 1]

# Plot scaled time trends
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_comparison, x='date', y='total_crimes_scaled', label='All Crimes (scaled)')
sns.lineplot(data=monthly_comparison, x='date', y='res_burglaries_scaled', label='Residential Burglaries (scaled)')
plt.title('Monthly Trend (Scaled): All Crimes vs. Residential Burglaries')
plt.xlabel('Date')
plt.ylabel('Scaled Crime Count (0–1)')
plt.legend()
plt.tight_layout()
plt.show()

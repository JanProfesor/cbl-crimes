import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load datasets
all_crimes = pd.read_csv('processed/final_dataset_all.csv')
res_burglary = pd.read_csv('processed/final_dataset_residential_burglary.csv')

# Create datetime column
all_crimes['date'] = pd.to_datetime(all_crimes[['year', 'month']].assign(day=1))
res_burglary['date'] = pd.to_datetime(res_burglary[['year', 'month']].assign(day=1))

# Aggregate monthly totals
monthly_all = all_crimes.groupby('date')['burglary_count'].sum().reset_index(name='total_crimes')
monthly_res = res_burglary.groupby('date')['burglary_count'].sum().reset_index(name='residential_burglaries')

# Set date as index for resampling
monthly_all.set_index('date', inplace=True)
monthly_res.set_index('date', inplace=True)

# Compute 3-month averages (calendar quarters)
quarterly_all = monthly_all['total_crimes'].resample('3MS').mean()
quarterly_res = monthly_res['residential_burglaries'].resample('3MS').mean()

# Combine into one DataFrame
quarterly_df = pd.concat([quarterly_all, quarterly_res], axis=1).dropna().reset_index()

# Scale counts 0–1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(quarterly_df[['total_crimes', 'residential_burglaries']])
quarterly_df['total_crimes_scaled'] = scaled[:, 0]
quarterly_df['res_burglaries_scaled'] = scaled[:, 1]

# Plot scaled quarterly trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=quarterly_df, x='date', y='total_crimes_scaled', label='All Crimes (scaled)')
sns.lineplot(data=quarterly_df, x='date', y='res_burglaries_scaled', label='Residential Burglaries (scaled)')
plt.title('Quarterly Average Trend (Scaled): All Crimes vs. Residential Burglaries')
plt.xlabel('Quarter Start Date')
plt.ylabel('Scaled Crime Count (0–1)')
plt.legend()
plt.tight_layout()
plt.show()

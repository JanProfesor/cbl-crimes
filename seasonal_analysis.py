import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('ward_london.csv')

# Create season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # [9, 10, 11]
        return 'Autumn'

# Add season column
df['season'] = df['month'].apply(get_season)

# Calculate seasonal statistics
seasonal_stats = df.groupby('season')['burglary_count'].agg(['mean', 'median', 'std', 'sum', 'count']).round(2)

# Create the plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Seasonal Burglary Analysis - London Wards', fontsize=16, fontweight='bold')

# 1. Average burglaries per season (bar plot)
seasons_order = ['Spring', 'Summer', 'Autumn', 'Winter']
colors = ['#2E8B57', '#FF6347', '#DAA520', '#4682B4']  # Green, Red, Gold, Blue

seasonal_means = seasonal_stats.loc[seasons_order, 'mean']
bars1 = ax1.bar(seasons_order, seasonal_means, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Average Burglaries per Ward-Month by Season', fontweight='bold')
ax1.set_ylabel('Average Burglary Count')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, seasonal_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# 2. Box plot showing distribution
box_data = [df[df['season'] == season]['burglary_count'] for season in seasons_order]
bp = ax2.boxplot(box_data, labels=seasons_order, patch_artist=True)
ax2.set_title('Burglary Distribution by Season', fontweight='bold')
ax2.set_ylabel('Burglary Count')
ax2.grid(axis='y', alpha=0.3)

# Color the box plots
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 3. Total burglaries per season (pie chart)
seasonal_totals = seasonal_stats.loc[seasons_order, 'sum']
ax3.pie(seasonal_totals, labels=seasons_order, colors=colors, autopct='%1.1f%%', 
        startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
ax3.set_title('Total Burglaries Distribution by Season', fontweight='bold')

# 4. Monthly trend line
monthly_avg = df.groupby('month')['burglary_count'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_colors = []
for month in range(1, 13):
    season = get_season(month)
    season_color_map = {'Spring': '#2E8B57', 'Summer': '#FF6347', 
                       'Autumn': '#DAA520', 'Winter': '#4682B4'}
    month_colors.append(season_color_map[season])

ax4.plot(range(1, 13), monthly_avg.values, marker='o', linewidth=2, markersize=8, color='black')
ax4.scatter(range(1, 13), monthly_avg.values, c=month_colors, s=100, alpha=0.8, edgecolors='black')
ax4.set_title('Monthly Burglary Trends', fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Average Burglary Count')
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(month_names, rotation=45)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Print seasonal statistics
print("\n" + "="*60)
print("SEASONAL BURGLARY STATISTICS")
print("="*60)
print(f"{'Season':<10} {'Mean':<8} {'Median':<8} {'Std Dev':<8} {'Total':<10} {'Records':<8}")
print("-"*60)
for season in seasons_order:
    stats = seasonal_stats.loc[season]
    print(f"{season:<10} {stats['mean']:<8.2f} {stats['median']:<8.2f} {stats['std']:<8.2f} {stats['sum']:<10.0f} {stats['count']:<8.0f}")

# Calculate percentage differences from average
overall_mean = df['burglary_count'].mean()
print(f"\nOverall average: {overall_mean:.2f} burglaries per ward-month")
print("\nSeasonal variations from average:")
for season in seasons_order:
    season_mean = seasonal_stats.loc[season, 'mean']
    pct_diff = ((season_mean - overall_mean) / overall_mean) * 100
    print(f"- {season}: {pct_diff:+.1f}% ({season_mean:.2f})")

# Statistical significance test
from scipy import stats
season_groups = [df[df['season'] == season]['burglary_count'] for season in seasons_order]
f_stat, p_value = stats.f_oneway(*season_groups)
print(f"\nANOVA Test for seasonal differences:")
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {p_value:.6f}")
if p_value < 0.05:
    print("✓ Statistically significant seasonal differences detected")
else:
    print("✗ No statistically significant seasonal differences")
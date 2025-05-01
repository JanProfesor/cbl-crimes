import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


base_folder = r"C:\Users\20234709\Documents\BDS Y2\Data Challenge 2 - Q4\Dataset"
csv_files = glob.glob(os.path.join(base_folder, "2024-*", "*.csv"))
csv_files.sort()

combined_df = pd.concat(
    (pd.read_csv(file) for file in csv_files),
    ignore_index=True
)

burglary_df = combined_df[combined_df['Crime type'] == 'Burglary']

output_file = os.path.join(base_folder, "combined-burglary-city-of-london-street.csv")
burglary_df.to_csv(output_file, index=False)

print(f"Filtered and saved {len(burglary_df)} rows where 'Crime type' is 'Burglary' into '{output_file}'.")



location_counts = burglary_df['Location'].value_counts()
repeat_locations = location_counts[location_counts >= 2]
print(f"Number of locations with 2 or more burglaries: {len(repeat_locations)}")



monthly_burglaries = burglary_df['Month'].value_counts().sort_index()
plt.figure(figsize=(10,6))
plt.plot(monthly_burglaries.index, monthly_burglaries.values, marker='o')
plt.title('Number of Burglaries per Month (2024)')
plt.xlabel('Month')
plt.ylabel('Number of Burglaries')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Load data
dataset = r"C:\Users\20234709\Documents\BDS Y2\Data Challenge 2 - Q4\Dataset\combined-burglary-city-of-london-street.csv"
df = pd.read_csv(dataset)

# Bar chart of Outcome type
outcome_counts = df['Last outcome category'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(x=outcome_counts.index, y=outcome_counts.values)
plt.title('Burglary Last Outcome Category')
plt.xlabel('Last Outcome Category')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




dataset = r"C:\Users\20234709\Documents\BDS Y2\Data Challenge 2 - Q4\Dataset\combined-burglary-city-of-london-street.csv"
df = pd.read_csv(dataset)

df_geo = df.dropna(subset=['Longitude', 'Latitude'])
fig = px.density_map(
    df_geo,
    lat='Latitude',
    lon='Longitude',
    z=None,
    radius=10,
    center=dict(lat=51.515, lon=-0.09),
    zoom=12,
    title="Burglary Hotspots - City of London",
    opacity=0.7,
    range_color=[0, 10],
)

fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white")
fig.show()
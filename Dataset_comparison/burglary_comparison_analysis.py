
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df_all = pd.read_csv("Dataset_comparison/final_dataset_all.csv")
df_res = pd.read_csv("Dataset_comparison/final_dataset_residential_burglary.csv")

df_all = df_all.rename(columns={"burglary_count": "all_crime_count"})
df_res = df_res.rename(columns={"burglary_count": "res_burglary_count"})

merged_df = df_all.merge(
    df_res[["ward_code", "year", "month", "res_burglary_count"]],
    on=["ward_code", "year", "month"]
)

merged_df["burglary_diff"] = merged_df["all_crime_count"] - merged_df["res_burglary_count"]
correlation = merged_df[["all_crime_count", "res_burglary_count"]].corr().iloc[0, 1]

print(f"Correlation between all and residential burglary counts: {correlation:.2f}")
print("\nSummary statistics of the difference (all - residential):")
print(merged_df["burglary_diff"].describe())

plt.figure(figsize=(8, 6))
plt.scatter(merged_df["all_crime_count"], merged_df["res_burglary_count"], alpha=0.4)
plt.title("Residential Burglary vs All Crime Count")
plt.xlabel("All Crime Count")
plt.ylabel("Residential Burglary Count")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(merged_df["burglary_diff"], bins=50, color='gray', edgecolor='black')
plt.title("Distribution of All Crime VS Residential Burglary")
plt.xlabel("Difference in Burglary Count")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

monthly_sum = merged_df.groupby(["year", "month"])[["all_crime_count", "res_burglary_count"]].sum().reset_index()
monthly_sum["date"] = pd.to_datetime(monthly_sum[["year", "month"]].assign(day=1))

plt.figure(figsize=(10, 6))
plt.plot(monthly_sum["date"], monthly_sum["all_crime_count"], label="All Crime")
plt.plot(monthly_sum["date"], monthly_sum["res_burglary_count"], label="Residential Burglary")
plt.title("Monthly Burglary Counts Over Time")
plt.xlabel("Date")
plt.ylabel("Burglary Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

X = merged_df["all_crime_count"].values.reshape(-1, 1)
y = merged_df["res_burglary_count"].values
model = LinearRegression().fit(X, y)
predicted = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.3, label='Data Points')
plt.plot(X, predicted, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Fit: All Crime vs Residential Burglary")
plt.xlabel("All Crime Count")
plt.ylabel("Residential Burglary Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

merged_df["res_percentage"] = (merged_df["res_burglary_count"] / merged_df["all_crime_count"]) * 100

plt.figure(figsize=(8, 6))
plt.hist(merged_df["res_percentage"].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title("Percentage of Residential Burglaries")
plt.xlabel("Residential % of Total Crime")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
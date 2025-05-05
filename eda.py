import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

final_df = pd.read_csv("processed/final_dataset.csv")

print("\n=== Shape ===")
print(final_df.shape)

print("\n=== Columns ===")
print(final_df.columns.tolist())

print("\n=== Data Types ===")
print(final_df.dtypes)

print("\n=== Summary Statistics ===")
print(final_df.describe(include="all"))

print("\n=== Missing Values ===")
missing = final_df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

if not missing.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.title("Missing Values per Column")
    plt.xlabel("Count")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

numeric_cols = final_df.select_dtypes(include="number")
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_cols.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(final_df["burglary_count"], bins=50, kde=True)
plt.title("Distribution of Burglary Count")
plt.xlabel("Burglary Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\n=== Average Monthly Burglary Count per Ward by Year ===")
print(final_df.groupby("year")["burglary_count"].mean())

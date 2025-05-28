# Final Dataset Description
#
# Each row represents a specific **ward** in London for a given month between Jan 2011 and Dec 2022.
#
# Columns:
# - ward_code: unique identifier for the electoral ward
# - year: extracted from date, representing the calendar year
# - month: numeric month value (1–12)
# - burglary_count: total number of reported burglaries in the ward that month
# - house_price: average house price in the ward that month (from LSOA-level forward/backward filled data)
# - crime_score: average IMD crime score across LSOAs in the ward
# - education_score: average IMD education/skills score across LSOAs in the ward
# - employment_score: average IMD employment score across LSOAs in the ward
# - environment_score: average IMD living environment score across LSOAs in the ward
# - health_score: average IMD health & disability score across LSOAs in the ward
# - housing_score: average IMD housing/services score across LSOAs in the ward
# - income_score: average IMD income score across LSOAs in the ward
# - avg_max_temperature: average daily max temperature for the month (°C)
# - max_temperature: highest recorded temperature in the month (°C)
# - min_max_temperature: lowest daily max temperature in the month (°C)
# - max_temperature_std: standard deviation of daily max temperatures (°C)
# - avg_min_temperature: average daily min temperature in the month (°C)
# - avg_temperature: average daily mean temperature in the month (°C)
# - total_rainfall: total rainfall in mm for the month
# - max_daily_rainfall: maximum rainfall in a single day that month (mm)
# - rainfall_std: standard deviation of daily rainfall amounts (mm)


import pandas as pd
import glob
from pathlib import Path


def load_burglary_data(filter_burglary):
    files = glob.glob("data/*-metropolitan-street.csv")
    records = []
    for file in files:
        df = pd.read_csv(file, usecols=["Month", "LSOA code", "Crime type"])
        if filter_burglary:
            df = df[df["Crime type"] == "Burglary"]
        df.rename(columns={"Month": "month", "LSOA code": "lsoa_code"}, inplace=True)
        df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M")
        records.append(df)
    df_all = pd.concat(records)
    return (
        df_all.groupby(["lsoa_code", "month"]).size().reset_index(name="burglary_count")
    )


def load_burglary_data_from_one_file(path):
    df = pd.read_csv(path, usecols=["Month", "LSOA code", "Crime type"])
    df.rename(columns={"Month": "month", "LSOA code": "lsoa_code"}, inplace=True)
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M")
    return df.groupby(["lsoa_code", "month"]).size().reset_index(name="burglary_count")


def load_raw_burglary_data():
    files = glob.glob("data/*-metropolitan-street.csv")
    records = []
    for file in files:
        records.append(pd.read_csv(file))
    df_all = pd.concat(records)
    return df_all


def load_house_prices(path):
    df = pd.read_csv(path)
    time_columns = [col for col in df.columns if col.startswith("Year ending")]
    df = df.melt(
        id_vars=["LSOA code"],
        value_vars=time_columns,
        var_name="raw_month",
        value_name="house_price",
    )
    df.rename(columns={"LSOA code": "lsoa_code"}, inplace=True)
    df["house_price"] = (
        df["house_price"].str.replace(",", "").str.replace('"', "").str.replace(":", "")
    )
    df["house_price"] = pd.to_numeric(
        df["house_price"], errors="coerce", downcast="unsigned"
    )
    df["month"] = pd.to_datetime(
        df["raw_month"].str.extract(r"Year ending (\w+ \d{4})")[0],
        format="%b %Y",
        errors="coerce",
    ).dt.to_period("M")
    df = df[df["month"].between("2011-01", "2022-12")]
    return df[["lsoa_code", "month", "house_price"]]


def load_weather_data(path):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df["TX"] *= 0.1
    df["TN"] *= 0.1
    df["TG"] *= 0.1
    df["RR"] *= 0.1
    df = (
        df.groupby("month")
        .agg(
            {
                "TX": ["mean", "max", "min", "std"],
                "TN": "mean",
                "TG": "mean",
                "RR": ["sum", "max", "std"],
            }
        )
        .reset_index()
    )
    df.columns = [
        "month",
        "avg_max_temperature",
        "max_temperature",
        "min_max_temperature",
        "max_temperature_std",
        "avg_min_temperature",
        "avg_temperature",
        "total_rainfall",
        "max_daily_rainfall",
        "rainfall_std",
    ]
    return df


def load_imd_scores(path, valid_range):
    df = pd.read_csv(path)
    df = df[df["Measurement"] == "Score"]
    domains = [
        "b. Income Deprivation Domain",
        "c. Employment Deprivation Domain",
        "d. Education, Skills and Training Domain",
        "e. Health Deprivation and Disability Domain",
        "f. Crime Domain",
        "g. Barriers to Housing and Services Domain",
        "h. Living Environment Deprivation Domain",
    ]
    df = df[df["Indices of Deprivation"].isin(domains)]
    df = df.pivot(
        index="FeatureCode", columns="Indices of Deprivation", values="Value"
    ).reset_index()
    df["valid_from"], df["valid_to"] = valid_range
    df.rename(columns={"FeatureCode": "lsoa_code"}, inplace=True)
    return df.rename(
        columns={
            "b. Income Deprivation Domain": "income_score",
            "c. Employment Deprivation Domain": "employment_score",
            "d. Education, Skills and Training Domain": "education_score",
            "e. Health Deprivation and Disability Domain": "health_score",
            "f. Crime Domain": "crime_score",
            "g. Barriers to Housing and Services Domain": "housing_score",
            "h. Living Environment Deprivation Domain": "environment_score",
        }
    )


def load_imd_2010_by_domain(folder):
    imd = None
    for path in glob.glob(f"{folder}/imd2010_*.csv"):
        domain = Path(path).stem.replace("imd2010_", "")
        df = pd.read_csv(path)
        df = df[df["Measurement"].str.contains("score", case=False, na=False)]
        df = df[["FeatureCode", "Value"]].rename(
            columns={"FeatureCode": "lsoa_code", "Value": f"{domain}_score"}
        )
        imd = df if imd is None else imd.merge(df, on="lsoa_code", how="outer")
    imd["valid_from"], imd["valid_to"] = 2011, 2014
    return imd


def build_dataset(crime_df, price_df, weather_df, imd_df):
    lookup = pd.read_csv("data/lsoa_to_ward_lookup_2020.csv")
    lsoas_in_lookup = set(lookup["LSOA11CD"])
    crime_df = crime_df[crime_df["lsoa_code"].isin(lsoas_in_lookup)]
    price_df = price_df[price_df["lsoa_code"].isin(lsoas_in_lookup)]
    imd_df = imd_df[imd_df["lsoa_code"].isin(lsoas_in_lookup)]

    all_months = pd.date_range("2011-01-01", "2022-12-01", freq="MS").to_period("M")
    lsoas = sorted(lsoas_in_lookup)
    full_index = pd.MultiIndex.from_product(
        [lsoas, all_months], names=["lsoa_code", "month"]
    )
    full_df = pd.DataFrame(index=full_index).reset_index()

    house_price_filled = full_df.merge(price_df, on=["lsoa_code", "month"], how="left")
    house_price_filled.sort_values(["lsoa_code", "month"], inplace=True)
    house_price_filled["house_price"] = (
        house_price_filled.groupby("lsoa_code")["house_price"].ffill().bfill()
    )

    df = house_price_filled.merge(crime_df, on=["lsoa_code", "month"], how="left")
    df["burglary_count"] = df["burglary_count"].fillna(0).astype(int)

    df["year"] = df["month"].dt.year
    df = df.merge(imd_df, on="lsoa_code", how="left")
    df = df[
        (df["valid_from"].isna())
        | ((df["year"] >= df["valid_from"]) & (df["year"] <= df["valid_to"]))
    ]
    df.drop(columns=["valid_from", "valid_to"], inplace=True)

    df = df.merge(weather_df, on="month", how="left")
    df["month"] = df["month"].dt.month

    df = df.merge(
        lookup[["LSOA11CD", "WD20CD"]],
        left_on="lsoa_code",
        right_on="LSOA11CD",
        how="left",
    )
    df = df.rename(columns={"WD20CD": "ward_code"}).drop(columns=["LSOA11CD"])

    ward_df = (
        df.groupby(["ward_code", "year", "month"])
        .agg(
            {
                "burglary_count": "sum",
                "house_price": "mean",
                "crime_score": "mean",
                "education_score": "mean",
                "employment_score": "mean",
                "environment_score": "mean",
                "health_score": "mean",
                "housing_score": "mean",
                "income_score": "mean",
                "avg_max_temperature": "mean",
                "max_temperature": "mean",
                "min_max_temperature": "mean",
                "max_temperature_std": "mean",
                "avg_min_temperature": "mean",
                "avg_temperature": "mean",
                "total_rainfall": "mean",
                "max_daily_rainfall": "mean",
                "rainfall_std": "mean",
            }
        )
        .reset_index()
    )

    imd_score_cols = [
        "crime_score",
        "education_score",
        "employment_score",
        "environment_score",
        "health_score",
        "housing_score",
        "income_score",
    ]
    ward_df[imd_score_cols] = ward_df[imd_score_cols].round(3)

    for col in ward_df.select_dtypes(include="float"):
        if col not in imd_score_cols:
            ward_df[col] = ward_df[col].round(2)

    return ward_df


def main():
    crime_all = load_burglary_data(False)
    crime_all_burglary = load_burglary_data(True)
    crime_raw = load_raw_burglary_data()
    crime_residential_burglary = load_burglary_data_from_one_file(
        "filtering_residential_burgary/residential-burglaries.csv"
    )
    crime_raw.to_csv("processed/burglary_data.csv", index=False)
    price = load_house_prices("data/median_price_1997-2023.csv")
    weather = load_weather_data("data/weather_2010-2022.csv")
    imd_2010 = load_imd_2010_by_domain("data")
    imd_2015 = load_imd_scores("data/imd2015lsoa.csv", (2015, 2018))
    imd_2019 = load_imd_scores("data/imd2019lsoa.csv", (2019, 2022))
    imd_all = pd.concat([imd_2010, imd_2015, imd_2019], ignore_index=True)

    ward_all_df = build_dataset(crime_all, price, weather, imd_all)
    ward_all_burglary_df = build_dataset(crime_all_burglary, price, weather, imd_all)
    ward__residential_burglary_df = build_dataset(
        crime_residential_burglary, price, weather, imd_all
    )

    ward_all_df.to_csv("processed/final_dataset_all.csv", index=False)
    ward_all_burglary_df.to_csv("processed/final_dataset_all_burglary.csv", index=False)
    ward__residential_burglary_df.to_csv(
        "processed/final_dataset_residential_burglary.csv", index=False
    )


if __name__ == "__main__":
    main()

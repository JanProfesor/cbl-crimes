import pandas as pd
from pathlib import Path
from itertools import product
from tqdm import tqdm

from data_services.crime_service import CrimeService
from data_services.imd_service import IMDService
from data_services.weather_service import WeatherService
from data_services.house_price_service import HousePriceService

WARD_MAP_PATH = Path("data/raw/lsoa_ward/lsoa_ward.csv")
OUTPUT_LONDON = Path("data/final/ward_london.csv")
OUTPUT_NON_LONDON = Path("data/final/ward_non_london.csv")


class DataJoiner:
    def __init__(self):
        self.weather_df = WeatherService().get_data()
        self.imd_df = IMDService().get_data()
        self.crime_df = CrimeService().get_data()
        self.price_df = HousePriceService().get_data()

        validate_ward_dataset(self.weather_df)
        validate_ward_dataset(self.imd_df)
        validate_ward_dataset(self.crime_df)
        validate_ward_dataset(self.price_df)

    def build(self):
        print("Merging all datasets...")

        merged = self.weather_df.merge(
            self.imd_df, on=["ward_code", "borough_code", "year", "month"], how="left"
        )
        merged = merged.merge(
            self.crime_df, on=["ward_code", "borough_code", "year", "month"], how="left"
        )
        merged = merged.merge(
            self.price_df, on=["ward_code", "borough_code", "year", "month"], how="left"
        )

        OUTPUT_LONDON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_NON_LONDON.parent.mkdir(parents=True, exist_ok=True)

        is_london = merged["borough_code"].str.startswith("E09", na=False)
        is_not_city_of_london = merged["borough_code"] != "E09000001"

        merged[is_london & is_not_city_of_london].drop(columns=["borough_code"]).to_csv(
            OUTPUT_LONDON, index=False
        )
        merged[~is_london & is_not_city_of_london].drop(
            columns=["borough_code"]
        ).to_csv(OUTPUT_NON_LONDON, index=False)

        validate_ward_dataset(merged)

        print(f"Saved {len(merged)} total rows.")
        print(f"London: {OUTPUT_LONDON}")
        print(f"Non-London: {OUTPUT_NON_LONDON}")


def validate_ward_dataset(df: pd.DataFrame) -> None:
    """
    Fast version of ward dataset validation:
    1. Ensures all expected ward_code + borough_code combinations exist.
    2. Ensures full coverage for each ward from 2011-01 to 2024-12.
    3. Ensures no missing values in data columns.
    """

    mapping_path = Path("data/raw/lsoa_ward/lsoa_ward.csv")
    mapping = pd.read_csv(mapping_path).rename(
        columns={
            "LSOA21CD": "lsoa_code",
            "WD24CD": "ward_code",
            "LAD24CD": "borough_code",
        }
    )
    expected_wards = mapping[["ward_code", "borough_code"]].drop_duplicates()

    actual_wards = df[["ward_code", "borough_code"]].drop_duplicates()
    expected_wards = expected_wards[expected_wards["ward_code"].str.startswith("E")]
    missing_wards = pd.merge(
        expected_wards,
        actual_wards,
        on=["ward_code", "borough_code"],
        how="left",
        indicator=True,
    )
    assert (missing_wards["_merge"] != "left_only").all(), "❌ Missing ward(s)"

    expected_months = pd.DataFrame(
        [(y, m) for y in range(2011, 2025) for m in range(1, 13)],
        columns=["year", "month"],
    )
    expected_months["key"] = 1
    expected_wards["key"] = 1
    expected_full = expected_wards.merge(expected_months, on="key").drop(columns="key")

    df_check = df[["ward_code", "borough_code", "year", "month"]].drop_duplicates()
    merged = expected_full.merge(
        df_check,
        on=["ward_code", "borough_code", "year", "month"],
        how="left",
        indicator=True,
    )
    assert (
        merged["_merge"] != "left_only"
    ).all(), "❌ Missing (ward, year, month) rows"

    id_cols = {"ward_code", "borough_code", "year", "month"}
    nulls = df.drop(columns=id_cols).isnull().sum()
    assert (nulls == 0).all(), f"❌ Nulls found in:\n{nulls[nulls > 0]}"

    print("✅ Dataset validation passed.")

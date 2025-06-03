import pandas as pd
from pathlib import Path


class CrimeService:
    RAW_BASE = Path("data/raw/crime")
    MAPPING_PATH = Path("data/raw/lsoa_ward/lsoa_ward.csv")
    PROCESSED_PATH = Path("data/processed/crime.csv")

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()
        self.__load()

    def __load(self):
        if self.PROCESSED_PATH.exists():
            print(f"Loading from cached file {self.PROCESSED_PATH}")
            self.data = pd.read_csv(self.PROCESSED_PATH)
            return

        print("Loading crime data from raw files...")

        mapping = pd.read_csv(self.MAPPING_PATH).rename(
            columns={
                "LSOA21CD": "lsoa_code",
                "WD24CD": "ward_code",
                "LAD24CD": "borough_code",
            }
        )
        mapping = mapping[["lsoa_code", "ward_code", "borough_code"]]

        all_files = list(self.RAW_BASE.glob("*/*/*-street.csv"))
        records = []

        for path in all_files:
            try:
                df: pd.DataFrame = pd.read_csv(
                    path,
                    usecols=["LSOA code", "Month", "Crime type"],
                    dtype=str,
                )
                df = df[df["Crime type"] == "Burglary"]
                df = df.rename(columns={"LSOA code": "lsoa_code", "Month": "month"})
                df = df.dropna(subset=["lsoa_code", "month"])

                dt: pd.Series = pd.to_datetime(df["month"], format="%Y-%m").dt
                df["year"] = dt.year
                df["month"] = dt.month

                records.append(df[["lsoa_code", "year", "month"]])
            except Exception as e:
                print(f"Failed to read {path.name}: {e}")

        burglary_df = pd.concat(records, ignore_index=True)

        burglary_df = (
            burglary_df.groupby(["lsoa_code", "year", "month"])
            .size()
            .reset_index(name="burglary_count")
        )

        merged = burglary_df.merge(mapping, on="lsoa_code", how="left")
        merged = (
            merged.groupby(["ward_code", "borough_code", "year", "month"])
            .agg(burglary_count=("burglary_count", "sum"))
            .reset_index()
        )

        all_wards = (
            mapping[["ward_code", "borough_code"]]
            .drop_duplicates()
            .query("ward_code.str.startswith('E')", engine="python")
        )

        periods = [
            (y, m)
            for y in range(2010, 2025)
            for m in range(1, 13)
            if not (y == 2010 and m < 12) and not (y == 2024 and m > 12)
        ]
        period_df = pd.DataFrame(periods, columns=["year", "month"])
        full_index = all_wards.merge(period_df, how="cross")

        full_df = full_index.merge(
            merged, on=["ward_code", "borough_code", "year", "month"], how="left"
        )
        full_df["burglary_count"] = full_df["burglary_count"].fillna(0)

        full_df = full_df.sort_values(by=["ward_code", "year", "month"])
        full_df["burglary_count_lag1"] = full_df.groupby("ward_code")[
            "burglary_count"
        ].shift(1)

        full_df = full_df[
            ~((full_df["year"] == 2010) & (full_df["month"] == 12))
        ].reset_index(drop=True)
        self.data = full_df
        self.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.PROCESSED_PATH, index=False)
        print(f"Saved to {self.PROCESSED_PATH}")

    def get_data(self) -> pd.DataFrame:
        if self.data.empty:
            raise RuntimeError("Crime data not loaded.")
        return self.data.copy()

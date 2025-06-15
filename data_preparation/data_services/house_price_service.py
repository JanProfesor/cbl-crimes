import pandas as pd
from pathlib import Path


class HousePriceService:
    LOCAL_XLS_PATH = Path("data/raw/house_prices/house_prices.xls")
    MAPPING_PATH = Path("data/raw/lsoa_ward/lsoa_ward.csv")
    PROCESSED_PATH = Path("data/processed/house_price.csv")

    def __init__(self) -> None:
        self.data: pd.DataFrame = pd.DataFrame()
        self.__load()

    def __load(self) -> None:
        if self.PROCESSED_PATH.exists():
            print(f"Loading processed house prices from {self.PROCESSED_PATH}")
            self.data = pd.read_csv(self.PROCESSED_PATH)
            return

        if not self.LOCAL_XLS_PATH.exists():
            raise FileNotFoundError(f"{self.LOCAL_XLS_PATH} does not exist.")

        print(f"Processing house prices from raw data: {self.LOCAL_XLS_PATH}")
        df = pd.read_excel(self.LOCAL_XLS_PATH, sheet_name="1a", skiprows=5)
        df.columns = [str(c).strip() for c in df.columns]

        time_columns = [col for col in df.columns if col.startswith("Year ending")]
        df[time_columns] = df[time_columns].apply(
            pd.to_numeric, errors="coerce", downcast="unsigned"
        )
        df[time_columns] = df[time_columns].ffill(axis=1)

        df = df.melt(
            id_vars=["LSOA code"],
            value_vars=time_columns,
            var_name="raw_month",
            value_name="house_price",
        )
        df.rename(columns={"LSOA code": "lsoa_code"}, inplace=True)

        df["month_period"] = pd.to_datetime(
            df["raw_month"].str.extract(r"Year ending (\w+ \d{4})")[0],
            format="%b %Y",
            errors="coerce",
        ).dt.to_period("M")

        df = df.dropna(subset=["house_price"])
        df["house_price"] = pd.to_numeric(
            df["house_price"], errors="coerce", downcast="unsigned"
        )
        df["year"] = df["month_period"].dt.year
        df["month"] = df["month_period"].dt.month

        mapping = pd.read_csv(self.MAPPING_PATH).rename(
            columns={
                "LSOA21CD": "lsoa_code",
                "WD24CD": "ward_code",
                "LAD24CD": "borough_code",
            }
        )
        mapping = mapping[["lsoa_code", "ward_code", "borough_code"]]

        merged = df.merge(mapping, on="lsoa_code", how="left")

        # Aggregate all LSOAs into average price per ward per month
        merged = (
            merged.groupby(["ward_code", "borough_code", "month_period"])["house_price"]
            .mean()
            .reset_index()
        )

        merged = merged[merged["month_period"].between("2010-12", "2024-12")]

        all_wards = (
            mapping[["ward_code", "borough_code"]]
            .drop_duplicates()
            .query("ward_code.str.startswith('E')", engine="python")
        )
        periods = pd.period_range(start="2011-01", end="2024-12", freq="M")
        period_df = pd.DataFrame(
            [(p.year, p.month, p) for p in periods],
            columns=["year", "month", "month_period"],
        )
        full_index = all_wards.merge(period_df, how="cross")

        full_df = full_index.merge(
            merged[["ward_code", "borough_code", "month_period", "house_price"]],
            on=["ward_code", "borough_code", "month_period"],
            how="left",
        )
        full_df["house_price"] = full_df.groupby("ward_code")["house_price"].transform(
            lambda x: x.ffill().bfill()
        )
        full_df["house_price"] = full_df.groupby("ward_code")["house_price"].transform(
            lambda grp: grp.fillna(grp.mean())
        )
        overall_mean = full_df["house_price"].mean(skipna=True)
        full_df["house_price"] = full_df["house_price"].fillna(overall_mean)

        self.data = full_df.sort_values(by=["ward_code", "year", "month"]).reset_index(
            drop=True
        )
        self.data = self.data.drop(columns=["month_period"])

        self.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.PROCESSED_PATH, index=False)
        print(f"Saved processed house prices to {self.PROCESSED_PATH}")

    def get_data(self) -> pd.DataFrame:
        if self.data.empty:
            raise RuntimeError("House price data not loaded.")
        return self.data.copy()

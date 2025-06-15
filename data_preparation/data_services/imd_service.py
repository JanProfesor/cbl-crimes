import pandas as pd
from pathlib import Path


class IMDService:
    RAW_DIR = Path("data/raw/imd")
    MAPPING_PATH = Path("data/raw/lsoa_ward/lsoa_ward.csv")
    PROCESSED_PATH = Path("data/processed/imd_scores.csv")

    DOMAIN_MAP_2010 = {
        "income": "imd-income-2010.csv",
        "employment": "imd-employment-2010.csv",
        "education": "imd-education-2010.csv",
        "health": "imd-health-2010.csv",
        "crime": "imd-crime-2010.csv",
        "housing": "imd-housing-2010.csv",
        "environment": "imd-environment-2010.csv",
    }

    DOMAIN_LABELS_2015_2019 = {
        "b. Income Deprivation Domain": "income",
        "c. Employment Deprivation Domain": "employment",
        "d. Education, Skills and Training Domain": "education",
        "e. Health Deprivation and Disability Domain": "health",
        "f. Crime Domain": "crime",
        "g. Barriers to Housing and Services Domain": "housing",
        "h. Living Environment Deprivation Domain": "environment",
    }

    RELEASE_YEARS = [2010, 2015, 2019]

    def __init__(self) -> None:
        self.data: pd.DataFrame = pd.DataFrame()
        self.__load()

    def __load(self) -> None:
        if self.PROCESSED_PATH.exists():
            print(f"Loading from cached file {self.PROCESSED_PATH}")
            self.data = pd.read_csv(self.PROCESSED_PATH)
            return

        print("Processing IMD data...")
        records = []

        for domain, filename in self.DOMAIN_MAP_2010.items():
            path = self.RAW_DIR / filename
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                records.append(
                    {
                        "lsoa_code": row["FeatureCode"],
                        "year": 2011,
                        "domain": domain,
                        "score": row["Value"],
                    }
                )

        for year in [2015, 2019]:
            path = self.RAW_DIR / f"imd-{year}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[
                (df["Measurement"] == "Score")
                & df["Indices of Deprivation"].isin(self.DOMAIN_LABELS_2015_2019)
            ]
            for _, row in df.iterrows():
                records.append(
                    {
                        "lsoa_code": row["FeatureCode"],
                        "year": year,
                        "domain": self.DOMAIN_LABELS_2015_2019[
                            row["Indices of Deprivation"]
                        ],
                        "score": row["Value"],
                    }
                )

        df = pd.DataFrame(records)
        df = df.pivot(
            index=["lsoa_code", "year"], columns="domain", values="score"
        ).reset_index()

        mapping = pd.read_csv(self.MAPPING_PATH).rename(
            columns={
                "LSOA21CD": "lsoa_code",
                "WD24CD": "ward_code",
                "LAD24CD": "borough_code",
            }
        )
        mapping = mapping[["lsoa_code", "ward_code", "borough_code"]]

        df = df.merge(mapping, on="lsoa_code", how="left")

        df = (
            df.groupby(["ward_code", "borough_code", "year"])
            .mean(numeric_only=True)
            .reset_index()
        )

        all_wards = (
            mapping[["ward_code", "borough_code"]]
            .drop_duplicates()
            .query("ward_code.str.startswith('E')", engine="python")
        )

        periods = [(y, m) for y in range(2011, 2025) for m in range(1, 13)]

        period_df = pd.DataFrame(periods, columns=["year", "month"])
        full_index = all_wards.merge(period_df, how="cross")

        df = df.sort_values(by=["ward_code", "year"])

        full_df = full_index.merge(
            df, on=["ward_code", "borough_code", "year"], how="left"
        )
        full_df = full_df.sort_values(by=["ward_code", "year", "month"])
        full_df = (
            full_df.sort_values(by=["ward_code", "year", "month"])
            .groupby("ward_code", group_keys=False)
            .apply(lambda g: g.ffill())
            .reset_index(drop=True)
        )
        full_df = (
            full_df.sort_values(by=["ward_code", "year", "month"])
            .groupby("ward_code", group_keys=False)
            .apply(lambda g: g.bfill())
            .reset_index(drop=True)
        )

        id_cols = {"ward_code", "borough_code", "year", "month"}
        value_cols = [c for c in full_df.columns if c not in id_cols]

        overall_means = full_df[value_cols].mean()
        full_df[value_cols] = full_df[value_cols].fillna(overall_means)

        self.data = full_df
        self.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.PROCESSED_PATH, index=False)
        print(f"Saved to {self.PROCESSED_PATH}")

    def get_data(self) -> pd.DataFrame:
        if self.data.empty:
            raise RuntimeError("IMD data not loaded.")
        return self.data.copy()

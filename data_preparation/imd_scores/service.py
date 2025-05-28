import pandas as pd
import requests
from io import StringIO

class IMDService:
    def __init__(self) -> None:
        self.data: dict[int, dict[str, float]] = {}
        self._load_all()

    def _load_all(self) -> None:
        self.data[2019] = self._load_csv(
            "https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd2019%2Findices",
            target_domains=[
                "Income Deprivation Domain",
                "Employment Deprivation Domain",
                "Education, Skills and Training Domain",
                "Health Deprivation and Disability Domain",
                "Crime Domain",
                "Barriers to Housing and Services Domain",
                "Living Environment Deprivation Domain"
            ]
        )

        self.data[2015] = self._load_csv(
            "https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd%2Findices",
            target_domains=[
                "Income Deprivation Domain",
                "Employment Deprivation Domain",
                "Education, Skills and Training Domain",
                "Health Deprivation and Disability Domain",
                "Crime Domain",
                "Barriers to Housing and Services Domain",
                "Living Environment Deprivation Domain"
            ]
        )

        self.data[2010] = self._load_2010_split()

    def _load_csv(self, url: str, target_domains: list[str]) -> dict[str, float]:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        df = df[df["Measurement"] == "Score"]
        df = df[df["Indices of Deprivation"].isin(target_domains)]

        result = {}
        for _, row in df.iterrows():
            lsoa = row["FeatureCode"]
            domain = row["Indices of Deprivation"]
            value = row["Value"]

            if lsoa not in result:
                result[lsoa] = {}
            result[lsoa][domain] = value

        return result

    def _load_2010_split(self) -> dict[str, float]:
        domain_map = {
            "Income": "Income Deprivation Domain",
            "Employment": "Employment Deprivation Domain",
            "Education": "Education, Skills and Training Domain",
            "Health": "Health Deprivation and Disability Domain",
            "Crime": "Crime Domain",
            "Housing": "Barriers to Housing and Services Domain",
            "Environment": "Living Environment Deprivation Domain"
        }

        base_url = "https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fdeprivation%2Fimd-{}-score-2010"
        result = {}

        for short, full in domain_map.items():
            url = base_url.format(short.lower())
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))

            for _, row in df.iterrows():
                lsoa = row["FeatureCode"]
                value = row["Value"]
                if lsoa not in result:
                    result[lsoa] = {}
                result[lsoa][full] = value

        return result

    def get_scores(self, lsoa_code: str, year: int) -> dict[str, float] | None:
        if year >= 2019:
            data_year = 2019
        elif year >= 2015:
            data_year = 2015
        else:
            data_year = 2010

        year_data = self.data.get(data_year, {})
        if not year_data:
            return None

        return year_data.get(lsoa_code, None)

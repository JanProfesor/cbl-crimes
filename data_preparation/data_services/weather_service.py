import pandas as pd
from math import radians, cos, sin, sqrt, atan2
from pathlib import Path
import re


class WeatherService:
    RAW_WEATHER_DIR = Path("data/raw/weather")
    RAW_LSOA_COORDS_PATH = Path("data/raw/lsoa_coords/lsoa_coords.csv")
    WARD_MAP_PATH = Path("data/raw/lsoa_ward/lsoa_ward.csv")
    PROCESSED_PATH = Path("data/processed/weather.csv")

    def __init__(self):
        self.data = pd.DataFrame()
        self.__load()

    def __load(self):
        if self.PROCESSED_PATH.exists():
            print(f"Loading from cached file {self.PROCESSED_PATH}")
            self.data = pd.read_csv(self.PROCESSED_PATH)
            return

        print("Loading and processing weather data...")
        lsoa_coords = pd.read_csv(self.RAW_LSOA_COORDS_PATH).rename(
            columns={"LSOA21CD": "lsoa_code", "Shape__Area": "area"}
        )
        ward_map = pd.read_csv(self.WARD_MAP_PATH).rename(
            columns={
                "LSOA21CD": "lsoa_code",
                "WD24CD": "ward_code",
                "LAD24CD": "borough_code",
            }
        )[["lsoa_code", "ward_code", "borough_code"]]

        station_records = []
        for path in self.RAW_WEATHER_DIR.glob("*.txt"):
            station_data = self.__parse_station(path)
            if not station_data:
                continue
            for entry in station_data["data"]:
                station_records.append(
                    {
                        "station": station_data["station"],
                        "lat": station_data["lat"],
                        "lon": station_data["lon"],
                        "year": entry["year"],
                        "month": entry["month"],
                        **{
                            k: v for k, v in entry.items() if k not in {"year", "month"}
                        },
                    }
                )

        weather_df = pd.DataFrame(station_records)
        weather_df["date"] = pd.to_datetime(weather_df[["year", "month"]].assign(day=1))
        weather_df = weather_df[
            (weather_df["date"] >= "2010-12-01") & (weather_df["date"] <= "2024-12-01")
        ].drop(columns=["date"])

        stations = weather_df[["station", "lat", "lon"]].drop_duplicates()

        lsoa_station_map = []
        for _, lsoa_row in lsoa_coords.iterrows():
            lat1, lon1 = float(lsoa_row["LAT"]), float(lsoa_row["LONG"])
            distances = stations.copy()
            distances["distance"] = distances.apply(
                lambda r: self.__haversine(lat1, lon1, r["lat"], r["lon"]), axis=1
            )
            nearest_station = distances.sort_values("distance").iloc[0]["station"]
            lsoa_station_map.append(
                {"lsoa_code": lsoa_row["lsoa_code"], "station": nearest_station}
            )

        lsoa_station_df = pd.DataFrame(lsoa_station_map)

        merged = lsoa_station_df.merge(weather_df, on="station", how="left")
        merged = merged.merge(ward_map, on="lsoa_code", how="left")

        periods = [(y, m) for y in range(2011, 2025) for m in range(1, 13)]
        all_wards = (
            ward_map[["ward_code", "borough_code"]]
            .drop_duplicates()
            .query("ward_code.str.startswith('E')", engine="python")
        )

        period_df = pd.DataFrame(periods, columns=["year", "month"])
        full_index = all_wards.merge(period_df, how="cross")

        ward_weather = (
            merged.groupby(["ward_code", "borough_code", "year", "month"])
            .agg(
                {
                    "tmax": "mean",
                    "tmin": "mean",
                    "af": "mean",
                    "rain": "mean",
                    "sun": "mean",
                }
            )
            .reset_index()
        )

        full_df = full_index.merge(
            ward_weather, on=["ward_code", "borough_code", "year", "month"], how="left"
        )

        for col in ["tmax", "tmin", "af", "rain", "sun"]:
            full_df[col] = full_df.groupby(["year", "month"])[col].transform(
                lambda x: x.fillna(x.mean())
            )

        self.data = full_df
        self.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.PROCESSED_PATH, index=False)
        print(f"Saved processed ward-level weather to {self.PROCESSED_PATH}")

    def get_data(self):
        if self.data.empty:
            raise RuntimeError("Weather data not loaded.")
        return self.data.copy()

    def __parse_station(self, path: Path) -> dict | None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None

        station, lat, lon = None, None, None
        for line in lines[:10]:
            if not station and re.match(r"^[A-Za-z]", line):
                station = line.strip()
            coord_match = re.search(r"Lat\s+([-\d.]+)\s+Lon\s+([-\d.]+)", line)
            if coord_match:
                lat, lon = float(coord_match[1]), float(coord_match[2])
        if not station or lat is None or lon is None:
            return None

        def clean(val, typ):
            if val == "---":
                return None
            try:
                return typ(val.strip("*#"))
            except ValueError:
                return None

        data = []
        for line in lines:
            if not re.match(r"^\s*\d{4}\s+\d{1,2}\s+", line):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            data.append(
                {
                    "year": clean(parts[0], int),
                    "month": clean(parts[1], int),
                    "tmax": clean(parts[2], float),
                    "tmin": clean(parts[3], float),
                    "af": clean(parts[4], int),
                    "rain": clean(parts[5], float),
                    "sun": clean(parts[6], float),
                }
            )

        return {"station": station, "lat": lat, "lon": lon, "data": data}

    def __haversine(self, lat1, lon1, lat2, lon2) -> float:
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

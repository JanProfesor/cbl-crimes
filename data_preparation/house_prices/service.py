import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

class HousePriceService:
    def __init__(self):
        self.data = None
        self.columns_by_quarter = []
        self._load_data()

    def _load_data(self):
        url = "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/housing/datasets/medianpricepaidbylowerlayersuperoutputareahpssadataset46/current/hpssadataset46medianpricepaidforresidentialpropertiesbylsoa.zip"
        response = requests.get(url)
        response.raise_for_status()

        with ZipFile(BytesIO(response.content)) as z:
            excel_file_name = next(name for name in z.namelist() if name.endswith(".xls"))
            with z.open(excel_file_name) as f:
                df = pd.read_excel(f, sheet_name="1a", skiprows=4)

        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={"LSOA code": "LSOA"})
        self.columns_by_quarter = [str(c) for c in df.columns if str(c).startswith("Year ending")]
        df["LSOA"] = df["LSOA"].astype(str).str.strip()
        self.data = df.set_index("LSOA")

    def get_price(self, lsoa_code: str, year: int, month: int) -> int | None:
        quarter_map = {
            1: "Mar", 2: "Mar", 3: "Mar",
            4: "Jun", 5: "Jun", 6: "Jun",
            7: "Sep", 8: "Sep", 9: "Sep",
            10: "Dec", 11: "Dec", 12: "Dec"
        }
        quarter = quarter_map.get(month)
        column = f"Year ending {quarter} {year}"
        if column not in self.columns_by_quarter:
            return None
        try:
            value = self.data.loc[lsoa_code, column]
            return int(value) if not pd.isna(value) else None
        except KeyError:
            return None

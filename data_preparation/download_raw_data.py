import requests
from pathlib import Path
from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO
import time
import sys
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

RAW_DIR = Path("data/raw")


def download_weather():
    base_url = "https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data"
    output_dir = RAW_DIR / "weather"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading weather data...")

    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.select_one("table.table.alternate-bg")
    links = table.find_all("a") if table else []

    links = [
        a["href"]
        for a in links
        if a.has_attr("href")
        and a["href"].endswith(".txt")
        and a["href"].startswith("http")
    ]

    for link in links:
        name = link.split("/")[-1].split(".")[0]
        output_path = output_dir / f"{name}.txt"

        r = requests.get(link, timeout=20)
        r.raise_for_status()
        output_path.write_text(r.text, encoding="utf-8")
        print(f"Saved: {output_path}")


def download_house_prices():
    url = (
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/housing/datasets/"
        "medianpricepaidbylowerlayersuperoutputareahpssadataset46/current/"
        "hpssadataset46medianpricepaidforresidentialpropertiesbylsoa.zip"
    )
    output_dir = RAW_DIR / "house_prices"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading house prices data...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with ZipFile(BytesIO(r.content)) as zf:
        original_name = next(name for name in zf.namelist() if name.endswith(".xls"))
        with zf.open(original_name) as source_file:
            target_path = output_dir / "house_prices.xls"
            with open(target_path, "wb") as out_file:
                out_file.write(source_file.read())

        print(f"Saved: {target_path}")


def download_imd():
    output_dir = RAW_DIR / "imd"
    output_dir.mkdir(parents=True, exist_ok=True)

    versions = {
        2010: [
            f"https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fdeprivation%2Fimd-{domain}-score-2010"
            for domain in [
                "income",
                "employment",
                "education",
                "health",
                "crime",
                "housing",
                "environment",
            ]
        ],
        2015: [
            "https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd%2Findices"
        ],
        2019: [
            "https://opendatacommunities.org/downloads/cube-table?uri=http%3A%2F%2Fopendatacommunities.org%2Fdata%2Fsocietal-wellbeing%2Fimd2019%2Findices"
        ],
    }

    print("Downloading IMD data...")

    for year, urls in versions.items():
        for url in urls:
            if year == 2010:
                domain = url.split("imd-")[1].split("-score")[0]
                filename = f"imd-{domain}-2010.csv"
            else:
                filename = f"imd-{year}.csv"

            dest = output_dir / filename

            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_text(r.text, encoding="utf-8")
            print(f"Saved: {dest}")


session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=2,
    allowed_methods=["GET"],
    status_forcelist=[502, 503, 504],
    raise_on_status=False,
)
session.mount("https://", HTTPAdapter(max_retries=retry))


def stream_with_resume(url: str, out_path: Path, chunk=1 << 20):
    done = out_path.exists() and out_path.stat().st_size or 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with session.get(url, headers=headers, stream=True, timeout=(10, 120)) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        total = (
            int(r.headers.get("Content-Range", "bytes */0").split("/")[-1])
            if "Content-Range" in r.headers
            else int(r.headers.get("Content-Length", 0))
        )
        mode = "ab" if done else "wb"
        start_time, bytes_so_far = time.time(), done
        with open(out_path, mode) as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if chunk_bytes:
                    f.write(chunk_bytes)
                    bytes_so_far += len(chunk_bytes)
                    speed = bytes_so_far / 1024 / max(time.time() - start_time, 0.1)
                    sys.stdout.write(
                        f"\r{out_path.name}: "
                        f"{bytes_so_far/1e6:,.1f} MB / {total/1e6:,.1f} MB "
                        f"({speed:,.0f} KB/s)"
                    )
                    sys.stdout.flush()
    print()


def download_crime():
    archives = ["2015-12", "2018-12", "2021-12", "2024-12"]
    base = Path("data/raw/crime")
    base.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        url = f"https://data.police.uk/data/archive/{archive}.zip"
        zip_path = base / f"{archive}.zip"
        target_folder = base / archive
        print(f"Downloading {archive}.zip")
        stream_with_resume(url, zip_path)

        print(f"\nExtracting {zip_path.name}")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(path=target_folder)

        for subfolder in target_folder.iterdir():
            if subfolder.is_dir():
                for file in subfolder.iterdir():
                    if file.is_file() and not file.name.endswith("-street.csv"):
                        file.unlink()

        zip_path.unlink()

        print(f"Done with {archive}\n")


def download_lsoa_coords():
    output_dir = RAW_DIR / "lsoa_coords"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_url = (
        "https://hub.arcgis.com/api/download/v1/items/"
        "68515293204e43ca8ab56fa13ae8a547/csv?redirect=false&layers=0"
    )
    print("Requesting download URL for LSOA coordinates...")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()

    try:
        signed_url = resp.json()["resultUrl"]
    except (KeyError, ValueError) as err:
        raise RuntimeError("Could not extract 'resultUrl' from API response") from err

    print("Downloading LSOA coordinates...")
    response = requests.get(signed_url, stream=True)
    response.raise_for_status()
    with open(output_dir / "lsoa_coords.csv", "wb") as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)


def download_lsoa_ward():
    output_dir = RAW_DIR / "lsoa_ward"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LSOA to Ward map...")

    url = "https://hub.arcgis.com/api/v3/datasets/686527814d73403e8f0a59c7a28b0c34_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_dir / "lsoa_ward.csv", "wb") as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)


def main():
    print("Starting raw data download...")
    download_weather()
    download_house_prices()
    download_imd()
    download_crime()
    download_lsoa_coords()
    download_lsoa_ward()
    print("All raw data downloaded.")


if __name__ == "__main__":
    main()

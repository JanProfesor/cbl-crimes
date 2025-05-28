import requests
from bs4 import BeautifulSoup
import re
from math import radians, cos, sin, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def extract_station_links(url: str) -> list[str]:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.select_one('table.table.alternate-bg')
    links = table.find_all('a') if table else []
    return [
        a['href'] for a in links
        if a.has_attr('href') and a['href'].endswith('.txt') and a['href'].startswith('http')
    ]


def parse_station(url: str) -> dict | None:
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.splitlines()

    station, lat, lon = None, None, None
    for line in lines[:10]:
        if not station and re.match(r'^[A-Za-z]', line):
            station = line.strip()
        coord_match = re.search(r'Lat\s+([-\d.]+)\s+Lon\s+([-\d.]+)', line)
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
        data.append({
            "year": clean(parts[0], int),
            "month": clean(parts[1], int),
            "tmax": clean(parts[2], float),
            "tmin": clean(parts[3], float),
            "af": clean(parts[4], int),
            "rain": clean(parts[5], float),
            "sun": clean(parts[6], float)
        })

    return {
        "station": station,
        "lat": lat,
        "lon": lon,
        "data": data
    }

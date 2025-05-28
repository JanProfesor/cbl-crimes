from .core import extract_station_links, parse_station, haversine

class WeatherService:
    def __init__(self):
        self.stations = []
        self._load_data()

    def _load_data(self):
        base_url = "https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data"
        print("Loading station links...")
        links = extract_station_links(base_url)
        print(f"Found {len(links)} stations. Parsing...")

        for link in links:
            parsed = parse_station(link)
            if parsed:
                self.stations.append(parsed)

        print(f"Loaded {len(self.stations)} stations.")

    def get_weather(self, lat: float, lon: float, year: int, month: int) -> dict | None:
        closest = min(self.stations, key=lambda s: haversine(lat, lon, s['lat'], s['lon']))
        for entry in closest['data']:
            if entry['year'] == year and entry['month'] == month:
                return {
                    "station": closest['station'],
                    "lat": closest['lat'],
                    "lon": closest['lon'],
                    "weather": entry
                }
        return None

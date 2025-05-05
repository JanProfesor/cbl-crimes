import folium
import json
import pandas as pd

df = pd.read_csv("processed/final_dataset.csv")

latest = df[(df["year"] == 2022) & (df["month"] == 12)]

with open("wards_2020_bsc_wgs84.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

m = folium.Map(location=[51.5, -0.1], zoom_start=10, tiles=None)

folium.Choropleth(
    geo_data=geojson_data,
    name="Burglary Count",
    data=latest,
    columns=["ward_code", "burglary_count"],
    key_on="feature.properties.WD20CD",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Burglary Count (Dec 2022)",
).add_to(m)

burglary_map = latest.set_index("ward_code")["burglary_count"].to_dict()

for feature in geojson_data["features"]:
    ward_code = feature["properties"]["WD20CD"]
    feature["properties"]["burglary_count"] = burglary_map.get(ward_code, 0)

folium.GeoJson(
    geojson_data,
    name="Ward Boundaries",
    tooltip=folium.GeoJsonTooltip(
        fields=["WD20CD", "WD20NM", "burglary_count"],
        aliases=["Ward Code:", "Ward Name:", "Burglary Count:"],
        localize=True,
    ),
    style_function=lambda _: {"fillOpacity": 0, "weight": 0},
).add_to(m)

m.save("interactive_map.html")

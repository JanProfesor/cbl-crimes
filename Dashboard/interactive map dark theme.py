import json
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
import branca.colormap as cm

# ─── 1. Load and optimize your CSV data ───────────────────────────
# Only load the columns we need to reduce memory usage
df = pd.read_csv("processed/final_dataset.csv", 
                 usecols=['year', 'month', 'ward_code', 'burglary_count'])

# Convert to appropriate data types to save memory
df['year'] = pd.to_numeric(df['year'], downcast='integer')
df['month'] = pd.to_numeric(df['month'], downcast='integer')
df['burglary_count'] = pd.to_numeric(df['burglary_count'], downcast='integer')

# Create date column more efficiently
df["date"] = pd.to_datetime(df[["year","month"]].assign(day=1))
df["timestr"] = df["date"].dt.strftime("%Y-%m-%d")

# ─── 2. Load and optimize your ward GeoJSON ────────────────────────
with open("wards_2020_bsc_wgs84.geojson", encoding="utf-8") as f:
    wards = json.load(f)

# Build a lookup from ward_code to geometry + name
# Only keep what we need from the geojson to save memory
ward_geoms = {}
for feat in wards["features"]:
    ward_code = feat["properties"]["WD20CD"]
    ward_geoms[ward_code] = {
        "geometry": feat["geometry"],
        "name": feat["properties"].get("WD20NM", "")
    }

# ─── 3. Build features for TimestampedGeoJson efficiently ────────────
# Determine min/max for color scaling
min_burglary, max_burglary = df["burglary_count"].min(), df["burglary_count"].max()

# Use a darker color palette better suited for dark backgrounds
colormap = cm.linear.Reds_09.scale(min_burglary, max_burglary)

# Process features more efficiently
ts_features = []
for _, row in df.iterrows():
    wc = row["ward_code"]
    if wc not in ward_geoms:
        continue
    
    ts_features.append({
        "type": "Feature",
        "geometry": ward_geoms[wc]["geometry"],
        "properties": {
            "time": row["timestr"],
            "style": {
                "color": "#444",  # Darker border color
                "weight": 0.5,
                "fillColor": colormap(row["burglary_count"]),
                "fillOpacity": 0.8
            },
            "popup": f"<div style='color:#333'><b>{ward_geoms[wc]['name']}</b><br/>{row['timestr']}: {row['burglary_count']} burglaries</div>"
        }
    })

ts_geojson = {"type": "FeatureCollection", "features": ts_features}

# ─── 4. Create the map with dark theme ────────────────────────────
m = folium.Map(
    location=[51.5074, -0.1278], 
    zoom_start=9,
    tiles="Cartodb dark_matter"  # Dark theme base map
)

# Add a legend
colormap.caption = 'Burglary Count'
colormap.add_to(m)

# Add optimized TimestampedGeoJson
time_slider = TimestampedGeoJson(
    ts_geojson,
    transition_time=200,
    period="P1M",
    add_last_point=False,
    auto_play=False,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY-MM",
    time_slider_drag_update=True,
    duration="P1M"  # Fixed duration helps performance
)
time_slider.add_to(m)

# Add custom CSS to adjust time slider to dark theme
custom_css = """
<style>
.leaflet-bar a {
    background-color: #222;
    color: #fff;
    border: 1px solid #444;
}
.leaflet-bar a:hover {
    background-color: #444;
}
.leaflet-popup-content-wrapper {
    background-color: #222;
    color: #ddd;
}
.leaflet-popup-tip {
    background-color: #222;
}
.slider {
    background-color: #333;
}
.slider .ui-slider-handle {
    background-color: #888;
    border: 1px solid #ccc;
}
/* Fix for white text on light slider */
.leaflet-control-layers-expanded, 
.leaflet-popup-content,
.slider-timestamp,
.leaflet-bar,
.leaflet-control-zoom-in,
.leaflet-control-zoom-out {
    color: #222 !important; /* Dark text color for better contrast */
}
/* Style the time display text specifically */
div.slider-timestamp {
    background-color: rgba(40, 40, 40, 0.8) !important;
    color: #fff !important;
    padding: 3px 6px;
    border-radius: 3px;
    border: 1px solid #555;
}
/* Style the time labels on the slider */
.leaflet-control-timecontrol .timecontrol-date {
    color: #fff !important;
    text-shadow: 1px 1px 1px rgba(0,0,0,0.8);
}
</style>
"""
m.get_root().html.add_child(folium.Element(custom_css))

# ─── 5. Save to HTML and optimize file ────────────────────────────
# Set width and height to fit viewport
m.get_root().width = "100%"
m.get_root().height = "100%"

# Save the optimized map
m.save("interactive_map_dark_theme.html")
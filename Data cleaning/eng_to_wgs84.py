import geopandas as gpd

gdf = gpd.read_file("wards_2020_bsc.geojson")

gdf = gdf.to_crs(epsg=4326)

gdf.to_file("wards_2020_bsc_wgs84.geojson", driver="GeoJSON")

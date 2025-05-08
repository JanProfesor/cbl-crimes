#!/usr/bin/env python3
import os
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union


def main():
    # 1) get the London boundary
    gdf_london = ox.geocode_to_gdf("London, UK")
    london_poly = gdf_london.loc[0, "geometry"]

    # 2) fetch residential land-use polygons
    tags_lu = {"landuse": "residential"}
    res_lu = ox.features_from_polygon(london_poly, tags_lu)
    res_lu = res_lu[res_lu.geom_type.isin(["Polygon", "MultiPolygon"])]

    # 3) fetch residential building footprints
    tags_bld = {"building": ["residential", "house", "apartments", "terrace", "detached", "semidetached"]}
    res_bld = ox.features_from_polygon(london_poly, tags_bld)
    res_bld = res_bld[res_bld.geom_type.isin(["Polygon", "MultiPolygon"])]

    # 4) union all residential geometries
    all_res_geoms = list(res_lu.geometry) + list(res_bld.geometry)
    residential_union = unary_union(all_res_geoms)
    res_mask = gpd.GeoDataFrame(geometry=[residential_union], crs="EPSG:4326")

    # 5) buffer the residential mask by 10 meters
    res_mask_m = res_mask.to_crs(epsg=3857)
    res_mask_m["geometry"] = res_mask_m.buffer(10)
    res_mask_buffered = res_mask_m.to_crs(epsg=4326)

    # 6) compute non-residential mask
    non_res_geom = london_poly.difference(res_mask_buffered.loc[0, "geometry"])
    non_res_mask = gpd.GeoDataFrame(geometry=[non_res_geom], crs="EPSG:4326")

    # 7) save masks as GeoJSON
    res_mask_buffered.to_file("london_residential_mask.geojson", driver="GeoJSON")
    non_res_mask.to_file("london_nonresidential_mask.geojson", driver="GeoJSON")

    # 8) plot and save overview map
    fig, ax = plt.subplots(figsize=(10, 10))
    non_res_mask.plot(ax=ax, color="lightgray", edgecolor="none")
    res_mask_buffered.plot(ax=ax, color="red", edgecolor="none", alpha=0.6)
    gdf_london.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
    ax.set_title("London: Residential (red) vs Non-Residential (gray) (buffered by 10m)")
    ax.axis("off")
    plt.savefig("london_residential_mask_buffered.png", dpi=300, bbox_inches="tight")

    print("Saved:")
    print(" • london_residential_mask.geojson (buffered)")
    print(" • london_nonresidential_mask.geojson")
    print(" • london_residential_mask_buffered.png")

if __name__ == "__main__":
    main()

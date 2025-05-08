#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point

def load_geojson(path):
    """
    Attempt to load a GeoJSON file into a GeoDataFrame.
    Try pyogrio (default), then Fiona, then manual JSON parsing.
    """
    # 1) default engine (pyogrio)
    try:
        return gpd.read_file(path)
    except Exception as e1:
        print(f"Warning: pyogrio failed to read {path}: {e1}")
    # 2) Fiona
    try:
        return gpd.read_file(path, engine="fiona")
    except Exception as e2:
        print(f"Warning: Fiona failed to read {path}: {e2}")
    # 3) manual JSON
    try:
        with open(path) as f:
            gj = json.load(f)
        features = gj.get('features') or [gj]
        geoms = []
        props = []
        for feat in features:
            geom = shape(feat['geometry'])
            geoms.append(geom)
            props.append(feat.get('properties', {}))
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs='EPSG:4326')
        return gdf
    except Exception as e3:
        print(f"Error: manual JSON parsing failed for {path}: {e3}")
        sys.exit(1)

def main():
    # base directory where this script and data files reside
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # expected filenames in base directory
    crime_filename = "2025-02-metropolitan-street.csv"
    mask_filename = "london_residential_mask.geojson"

    # full paths
    crime_csv = os.path.join(base_dir, crime_filename)
    mask_file = os.path.join(base_dir, mask_filename)

    # verify inputs exist
    if not os.path.exists(crime_csv):
        print(f" Crime CSV not found: {crime_csv}")
        sys.exit(1)
    if not os.path.exists(mask_file):
        print(f" Residential mask not found: {mask_file}")
        print("   Please run residential_mask.py to generate the mask.")
        sys.exit(1)

    # load and filter burglary records
    df = pd.read_csv(crime_csv)
    # adjust column name if necessary
    if 'Crime type' in df.columns:
        mask = df['Crime type'].str.contains('burglary', case=False, na=False)
    else:
        mask = df.iloc[:, 0].str.contains('burglary', case=False, na=False)
    df = df[mask].copy()
    if df.empty:
        print("No burglary records found in the CSV.")
        return

    # create GeoDataFrame of points
    df['geometry'] = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

    # load residential mask polygon
    res_mask = load_geojson(mask_file).to_crs(epsg=4326)
    res_poly = res_mask.geometry.unary_union  # union all parts

    # classify burglaries
    gdf['residential'] = gdf.geometry.within(res_poly)

    # split into residential and non-residential
    res_burglary = gdf[gdf.residential].copy()
    nonres_burglary = gdf[~gdf.residential].copy()

    # compute stats
    total = len(gdf)
    res_count = len(res_burglary)
    pct = (res_count / total * 100) if total > 0 else 0
    print(f"Residential burglaries: {res_count}/{total} ({pct:.1f}%)")

    # output file paths
    out_res_csv    = os.path.join(base_dir, '2025-02-residential-burglaries.csv')
    out_nonres_csv = os.path.join(base_dir, '2025-02-nonresidential-burglaries.csv')
    out_res_geo    = os.path.join(base_dir, '2025-02-residential-burglaries.geojson')
    out_nonres_geo = os.path.join(base_dir, '2025-02-nonresidential-burglaries.geojson')

    # save outputs
    res_burglary.to_csv(out_res_csv, index=False)
    nonres_burglary.to_csv(out_nonres_csv, index=False)
    res_burglary.to_file(out_res_geo, driver='GeoJSON')
    nonres_burglary.to_file(out_nonres_geo, driver='GeoJSON')

    print("Saved outputs:")
    print(f" • {out_res_csv}\n • {out_nonres_csv}\n • {out_res_geo}\n • {out_nonres_geo}")

if __name__ == '__main__':
    main()

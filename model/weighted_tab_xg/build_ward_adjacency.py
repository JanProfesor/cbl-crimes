import fiona
import pandas as pd
from shapely.geometry import shape

def build_ward_adjacency(shapefile_path: str, output_csv: str = "ward_adjacency.csv"):
    """
    Reads a London ward shapefile, computes which wards touch each other,
    and writes a CSV with columns [ward_code, neighbor_code].
    
    Assumes the shapefile has a unique ward identifier under 'GSS_CODE'.
    """
    # 1) Read shapefile via Fiona into a list of (ward_id, geometry) tuples
    ward_geoms = []
    with fiona.open(shapefile_path) as src:
        for rec in src:
            props = rec["properties"]
            geom = shape(rec["geometry"])
            ward_id = str(props["GSS_CODE"])
            ward_geoms.append((ward_id, geom))
    
    # 2) Compare each pair to see if they "touch" (share a boundary)
    adjacency_list = []
    n = len(ward_geoms)
    for i in range(n):
        id_i, geom_i = ward_geoms[i]
        for j in range(i + 1, n):
            id_j, geom_j = ward_geoms[j]
            if geom_i.touches(geom_j):
                adjacency_list.append((id_i, id_j))
                adjacency_list.append((id_j, id_i))
    
    # 3) Build DataFrame and save to CSV
    adj_df = pd.DataFrame(adjacency_list, columns=["ward_code", "neighbor_code"])
    adj_df = adj_df.sort_values(["ward_code", "neighbor_code"]).reset_index(drop=True)
    adj_df.to_csv(output_csv, index=False)
    print(f"Saved adjacency CSV to {output_csv} ({len(adj_df)} rows)")

if __name__ == "__main__":
    # Path to your shapefile (ensure .shp, .shx, .dbf are all present)
    shapefile_path = "adjacency/London_Ward_CityMerged.shp"
    build_ward_adjacency(shapefile_path, output_csv="ward_adjacency.csv")

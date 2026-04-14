import geopandas as gpd
import os

try:
    shp_path = 'ev_placement/delhi_administrative.shp'
    if not os.path.exists(shp_path):
        # Try alternate path logic from notebook_utils
        shp_path = 'delhi_administrative.shp'
    
    if os.path.exists(shp_path):
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
            
        bounds = gdf.total_bounds
        print(f"BOUNDS: {bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}") # minx, miny, maxx, maxy
    else:
        print("Shapefile not found.")

except Exception as e:
    print(f"Error: {e}")

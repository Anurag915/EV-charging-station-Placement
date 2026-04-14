import os
import sys

# Output file for results
output_file = "inspection_results_utf8.txt"

def log(msg):
    print(msg)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear previous output
if os.path.exists(output_file):
    os.remove(output_file)

try:
    import geopandas as gpd
    log("Geopandas imported successfully.")
except ImportError:
    log("Geopandas not installed.")
    sys.exit(1)

# Updated path based on user input
shapefile_path = os.path.join("Land_use_mrg", "Land_use_mrg", "Merged_Delhi.shp")

if not os.path.exists(shapefile_path):
    log(f"File not found: {shapefile_path}")
    # Try absolute path execution context check
    log(f"Current working directory: {os.getcwd()}")
    log(f"Directory listing of Land_use_mrg/Land_use_mrg:")
    try:
        log(str(os.listdir(os.path.join("Land_use_mrg", "Land_use_mrg"))))
    except Exception as e:
        log(f"Could not list directory: {e}")
    sys.exit(1)

log(f"File found: {shapefile_path}")

try:
    gdf = gpd.read_file(shapefile_path)
    log(f"CRS: {gdf.crs}")
    log(f"Columns: {gdf.columns.tolist()}")
    log("First 5 rows:")
    log(str(gdf.head()))
    
    log("\n--- Unique Values Analysis ---")
    
    # Check all object/string columns for potential land use classification
    potential_columns = [col for col in gdf.columns if gdf[col].dtype == 'object']
    
    for col in potential_columns:
        unique_vals = gdf[col].unique()
        count = len(unique_vals)
        log(f"\nColumn: '{col}' ({count} unique values)")
        if count < 50:
            log(f"Values: {unique_vals}")
        else:
            log(f"First 20 values: {unique_vals[:20]}")

except Exception as e:
    log(f"Error inspecting shapefile: {e}")
    import traceback
    with open(output_file, "a", encoding="utf-8") as f:
        traceback.print_exc(file=f)

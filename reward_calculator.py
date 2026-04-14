import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import json
import math
from data_generator import DataGenerator

class RewardCalculator:
    def __init__(self, data_dir='ev_placement'):
        self.data_dir = data_dir
        self.grid_shape = (50, 50)
        
        # Bounds (Delhi approx) - Defined early for use in sub-methods
        self.bounds = [76.80, 28.40, 77.35, 28.90] 
        self.minx, self.miny, self.maxx, self.maxy = self.bounds

        
        # --- 1. DATA LOADING ---
        # Generate Image-Based Data (Traffic only, Land Use comes from SHP now)
        gen = DataGenerator(self.grid_shape)
        self.traffic = gen.generate_traffic()
        
        # Initialize Land Use
        self.land_use = np.zeros(self.grid_shape) + 6 # Default to Neutral
        
        # Load Real Land Use from Shapefile
        shapefile_path = os.path.join(r"C:\Users\anura\OneDrive\Documents\Delhi Implementation\Land_use_mrg\Land_use_mrg", "Merged_Delhi.shp")
        if os.path.exists(shapefile_path):
            print(f"Loading land use from: {shapefile_path}")
            self._process_real_land_use(shapefile_path)
        else:
            print("Shapefile not found, using generative fallback.")
            self.land_use = gen.generate_land_use()
        
        # Load REAL Existing Stations
        self.existing_stations = []
        try:
            with open(os.path.join(data_dir, 'real_stations_delhi.json'), 'r') as f:
                data = json.load(f)
                for item in data:
                    self.existing_stations.append((item['lat'], item['lon']))
        except:
             # Fallback
             self.existing_stations = [(28.6139, 77.2090)]

    def _process_real_land_use(self, shapefile_path):
        try:
            gdf = gpd.read_file(shapefile_path)
            # Ensure we are in Lat/Lon
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Create a list of points for our grid
            points = []
            indices = []
            
            # Using the bounds defined in the class
            minx, miny, maxx, maxy = self.minx, self.miny, self.maxx, self.maxy
            
            for row in range(self.grid_shape[0]):
                for col in range(self.grid_shape[1]):
                    # Calculate lat/lon for center of cell
                    y_frac = (row + 0.5) / self.grid_shape[0]
                    x_frac = (col + 0.5) / self.grid_shape[1]
                    lat = maxy - y_frac * (maxy - miny)
                    lon = minx + x_frac * (maxx - minx)
                    
                    points.append(Point(lon, lat))
                    indices.append((row, col))
            
            # Create GeoDataFrame from grid points
            grid_gdf = gpd.GeoDataFrame({'geometry': points}, index=range(len(points)), crs="EPSG:4326")
            
            # Spatial Join
            # op='within' checks if point is within polygon
            joined = gpd.sjoin(grid_gdf, gdf, how="left", predicate="within")
            
            # Process results
            for idx, row_data in joined.iterrows():
                if pd.isna(row_data['layer']):
                    continue
                
                grid_row, grid_col = indices[idx]
                layer_name = str(row_data['layer']).lower()
                
                # Mapping Logic
                # 0: Green, 1: Water, 2: Res, 3: Comm/PSP, 5: Ind, 6: Neutral
                code = 6
                if 'green' in layer_name or 'water' in layer_name:
                    code = 0
                elif 'industrial' in layer_name:
                    code = 5
                elif 'psp' in layer_name: # Public Semi Public -> Commercial-ish
                    code = 3
                elif 'residential' in layer_name:
                    code = 2
                elif 'builtup' in layer_name: # Treat as residential/mixed
                    code = 2
                
                self.land_use[grid_row, grid_col] = code
                
            print("Successfully processed real land use data.")
            
        except Exception as e:
            print(f"Error processing shapefile: {e}")
            print("Reverting to neutral land use.")


        # Bounds (Delhi approx)
        self.bounds = [76.80, 28.40, 77.35, 28.90] 
        self.minx, self.miny, self.maxx, self.maxy = self.bounds
            
    def geo_to_grid(self, lat, lon):
        y_frac = (self.maxy - lat) / (self.maxy - self.miny)
        x_frac = (lon - self.minx) / (self.maxx - self.minx)
        
        row = int(y_frac * self.grid_shape[0])
        col = int(x_frac * self.grid_shape[1])
        
        row = max(0, min(row, self.grid_shape[0] - 1))
        col = max(0, min(col, self.grid_shape[1] - 1))
        return row, col
    
    def grid_to_geo(self, row, col):
        y_frac = (row + 0.5) / self.grid_shape[0]
        x_frac = (col + 0.5) / self.grid_shape[1]
        lat = self.maxy - y_frac * (self.maxy - self.miny)
        lon = self.minx + x_frac * (self.maxx - self.minx)
        return lon, lat

    def get_distance_km(self, lat1, lon1, lat2, lon2):
        R = 6371 
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def assess_location(self, lat, lon, new_stations=[]):
        row, col = self.geo_to_grid(lat, lon)
        
        # --- R_Infra ---
        min_dist_existing = float('inf')
        for s_lat, s_lon in self.existing_stations:
            d = self.get_distance_km(lat, lon, s_lat, s_lon)
            if d < min_dist_existing: min_dist_existing = d
        
        capped_dist = min(min_dist_existing, 5.0)
        r_infra = (capped_dist / 5.0) * 15.0
        
        # --- R_Traffic ---
        # +20 max
        val_traffic = self.traffic[row, col]
        r_traffic = val_traffic * 20.0
        
        # --- R_Land (Image Based) ---
        # 0:Green, 1:Water, 2:Res, 3:Comm, 5:Ind, 6:Neutral
        code = self.land_use[row, col]
        
        r_land = 0.0
        reason = "Neutral/Unknown"
        
        if code == 0:
            r_land = -50.0; reason = "Penalty: Green Belt (Protected)"
        elif code == 1:
            r_land = -100.0; reason = "Penalty: Water Body"
        elif code == 2:
            r_land = 10.0; reason = "Zone: Residential (Accessible)"
        elif code == 3:
            r_land = 30.0; reason = "Zone: Commercial/Market (High Demand)"
        elif code == 5:
            r_land = 15.0; reason = "Zone: Industrial"
        elif code == 6:
            r_land = -5.0; reason = "Zone: Unclassified/Neutral"
            
        # --- R_Crowding ---
        # --- R_Crowding ---
        r_crowding = 0.0
        
        # Combine new stations and existing stations for crowding check
        # We need to guard against crowding from BOTH layers
        all_stations = new_stations + self.existing_stations
        
        for n_lat, n_lon in all_stations:
            # Skip self-comparison if the point is exactly the same (e.g. updating an existing point)
            if abs(n_lat-lat) < 1e-5 and abs(n_lon-lon) < 1e-5: continue
            
            d = self.get_distance_km(lat, lon, n_lat, n_lon)
            if d < 0.5: # 500m radius
                r_crowding = -50.0
                break
        
        total_score = r_infra + r_traffic + r_land + r_crowding
        
        return {
            'R_Infra': round(r_infra, 2),
            'R_Traffic': round(r_traffic, 2),
            'R_Land': round(r_land, 2),
            'R_Crowding': round(r_crowding, 2),
            'Total': round(total_score, 2),
            'Land_Code': int(code),
            'Land_Reason': reason,
            'Nearest_Existing_Km': round(min_dist_existing, 2)
        }

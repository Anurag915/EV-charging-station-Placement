import numpy as np
import os
from reward_calculator import RewardCalculator

def sync_data():
    print("Initializing RewardCalculator to load shapefile data...")
    rc = RewardCalculator()
    
    # Ensure output directory exists
    os.makedirs('ev_placement', exist_ok=True)
    
    # 1. Land Use
    print("Saving landuse_r1.npy...")
    np.save('ev_placement/landuse_r1.npy', rc.land_use)
    print(f"Land use shape: {rc.land_use.shape}")
    print(f"Unique values: {np.unique(rc.land_use)}")
    
    # 2. Demand/Traffic (Reuse traffic as demand for now as per previous logic)
    print("Saving demand_avg.npy...")
    np.save('ev_placement/demand_avg.npy', rc.traffic)
    print(f"Demand shape: {rc.traffic.shape}")
    
    # 3. Create dummy stations mask/distance if they don't exist or just overwrite them to be clean
    # The training script expects these. Let's create empty ones to start fresh training.
    print("Resetting stations_mask.npy and stations_distance.npy...")
    stations_mask = np.zeros(rc.grid_shape)
    np.save('ev_placement/stations_mask.npy', stations_mask)
    
    stations_distance = np.full(rc.grid_shape, 100.0) # Far away
    np.save('ev_placement/stations_distance.npy', stations_distance)
    
    print("✅ Data sync complete. Ready for training.")

if __name__ == "__main__":
    sync_data()

import numpy as np
import matplotlib.pyplot as plt
from reward_calculator import RewardCalculator
import os

def verify():
    print("Initializing RewardCalculator...")
    rc = RewardCalculator()
    
    land_use = rc.land_use
    print(f"Land Use Shape: {land_use.shape}")
    
    unique, counts = np.unique(land_use, return_counts=True)
    print("Land Use Codes and Counts:")
    for u, c in zip(unique, counts):
        print(f"Code {int(u)}: {c} cells")
        
    # Check if we have meaningful data (not just all 6s)
    if len(unique) <= 1 and unique[0] == 6:
        print("FAILURE: Land use grid is entirely default/neutral (6). Shapefile integration failed.")
        # Print first few lines of log if any?
    else:
        print("SUCCESS: Land use grid contains mixed values.")
        
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(land_use, cmap='tab10', origin='upper') # simple colormap
    plt.colorbar(label='Land Use Code')
    plt.title("Integrated Land Use Grid (50x50)")
    output_path = "land_use_verification.png"
    plt.savefig(output_path)
    print(f"Map saved to {output_path}")

if __name__ == "__main__":
    verify()

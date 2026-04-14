import numpy as np

import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load data
root = Path(r'd:\Delhi Implementation\ev_placement')
demand_avg = np.load(root / 'demand_avg.npy')
stations = np.load(root / 'stations_mask.npy')
landuse = np.load(root / 'landuse_r1.npy')

with open(root / 'raster_meta.json', 'r') as f:
    meta = json.load(f)

minx, miny, maxx, maxy = meta['bounds']
res = meta['resolution_m']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Demand with stations
im1 = ax1.imshow(demand_avg, cmap='viridis', origin='upper',
                 extent=[minx, maxx, miny, maxy])
plt.colorbar(im1, ax=ax1, label='Average Probe Count')

# Add stations as cyan dots
station_y, station_x = np.where(stations > 0)
station_coords_x = minx + (station_x + 0.5) * res
station_coords_y = maxy - (station_y + 0.5) * res
ax1.scatter(station_coords_x, station_coords_y, c='cyan', s=50, 
            label='Existing Stations', edgecolor='white')
ax1.set_title('Traffic Demand and Existing Stations')
ax1.legend()

# Plot 2: Landuse suitability
im2 = ax2.imshow(landuse, cmap='RdYlGn', origin='upper',
                 extent=[minx, maxx, miny, maxy])
plt.colorbar(im2, ax=ax2, label='Landuse Suitability (R1)')
ax2.set_title('Landuse Suitability')

# Save with high DPI
plt.tight_layout()
output_path = root / 'demand_and_landuse_preview.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Saved visualization to {output_path}')

# Print some statistics
print('\nStatistics:')
print(f'Grid shape: {demand_avg.shape}')
print(f'Number of existing stations: {stations.sum()}')
print(f'Demand range: {demand_avg.min():.2f} to {demand_avg.max():.2f}')
print(f'Average demand: {demand_avg.mean():.2f}')
print(f'Non-zero cells: {(demand_avg > 0).sum()} of {demand_avg.size}')
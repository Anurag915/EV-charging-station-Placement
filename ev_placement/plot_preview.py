import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path(r'd:/Delhi Implementation/ev_placement')
meta=json.load(open(p/'raster_meta.json'))
minx,miny,maxx,maxy=meta['bounds']
res=meta['resolution_m']
height, width = meta['height'], meta['width']
arr=np.load(p/'demand_avg.npy')
st=np.load(p/'stations_mask.npy')
# build extent
extent=(minx, maxx, miny, maxy)
fig,ax=plt.subplots(figsize=(8,6))
im=ax.imshow(arr, cmap='inferno', extent=extent, origin='upper')
# overlay stations
px,py = np.where(st==1)
# get coords
coords_x = minx + (py + 0.5)*res
coords_y = maxy - (px + 0.5)*res
ax.scatter(coords_x, coords_y, c='cyan', s=10, edgecolor='k')
ax.set_title('Demand (avg probe count) with existing stations')
plt.colorbar(im, ax=ax, label='avg_probe_count')
plt.tight_layout()
out = p/'demand_preview.png'
plt.savefig(out, dpi=150)
print('Saved', out)

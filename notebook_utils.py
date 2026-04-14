"""
Notebook utility functions for the Delhi EV charging project.
Place common operations here to improve notebook readability and reuse.
"""
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


def load_delhi_boundary(shp_path: str = None):
    """Load Delhi administrative boundary shapefile from given path or default workspace file.
    Returns a GeoDataFrame.
    """
    if shp_path is None:
        shp_path = os.path.join(os.path.dirname(__file__), 'delhi_administrative.shp')
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        try:
            gdf = gdf.set_crs(epsg=4326)
        except Exception:
            pass
    return gdf


def get_stations_from_globals(env=None, agent=None, max_stations=120, saved_path=None):
    """Try multiple methods to obtain station placements:
    - in-memory `stations` variable if present
    - `stations_gdf` if present
    - a saved stations.npy file
    - run the agent on the env if available
    Returns a list of (x, y) tuples.
    """
    stations_list = []
    # 1) global variable 'stations'
    try:
        g = globals()
        if 'stations' in g and g['stations']:
            stations_list = [(float(x), float(y)) for x, y in g['stations']]
            return stations_list
    except Exception:
        pass
    # 2) stations_gdf
    try:
        if 'stations_gdf' in globals() and globals()['stations_gdf'] is not None:
            sg = globals()['stations_gdf']
            stations_list = [(float(pt.x), float(pt.y)) for pt in sg.geometry]
            return stations_list
    except Exception:
        pass
    # 3) saved numpy
    if saved_path is None:
        saved_path = os.path.join(os.getcwd(), 'visualization_outputs', 'stations.npy')
    if os.path.exists(saved_path):
        try:
            loaded = np.load(saved_path, allow_pickle=True)
            return [tuple(map(float, s)) for s in loaded]
        except Exception:
            pass
    # 4) derive from agent+env
    if env is not None and agent is not None:
        try:
            state = env.reset()
            for _ in range(int(max_stations)):
                action = agent.get_action(state)
                # handle different return types
                if isinstance(action, tuple):
                    a = action[0]
                else:
                    a = action
                try:
                    # if torch tensor
                    import torch
                    if isinstance(a, torch.Tensor):
                        a = a.detach().cpu().numpy()
                except Exception:
                    pass
                a = np.asarray(a)
                if a.size >= 2:
                    stations_list.append((float(a[0]), float(a[1])))
                try:
                    state, _, done, _ = env.step(a)
                except Exception:
                    pass
                if len(stations_list) >= max_stations:
                    break
            if stations_list:
                return stations_list
        except Exception:
            pass
    # nothing found
    return []


def save_stations(stations_list, out_dir='visualization_outputs'):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'stations.npy'), np.array(stations_list, dtype=object))


def compute_ward_coverage(delhi_boundary_gdf, stations_list, coverage_decay=2000.0):
    """Compute per-ward station counts, distance to nearest station (meters) and coverage score.
    Returns GeoDataFrame (projected to EPSG:3857) with columns station_count, dist_to_nearest_m, coverage.
    """
    if not len(stations_list):
        raise ValueError('stations_list must be non-empty')
    wards = delhi_boundary_gdf.copy()
    if wards.crs is None:
        wards = wards.set_crs(epsg=4326)
    wards_proj = wards.to_crs(epsg=3857)
    stations_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in stations_list], crs=wards.crs)
    stations_proj = stations_gdf.to_crs(epsg=3857)
    try:
        joined = gpd.sjoin(stations_proj, wards_proj, predicate='within', how='left')
    except TypeError:
        joined = gpd.sjoin(stations_proj, wards_proj, op='within', how='left')
    counts = joined.groupby('index_right').size()
    wards_proj['station_count'] = counts.reindex(wards_proj.index).fillna(0).astype(int)
    centroids = wards_proj.geometry.centroid
    nearest_dist = []
    for c in centroids:
        dists = stations_proj.geometry.distance(c)
        nearest_dist.append(dists.min() if len(dists) else np.nan)
    wards_proj['dist_to_nearest_m'] = nearest_dist
    wards_proj['coverage'] = np.exp(-wards_proj['dist_to_nearest_m'] / float(coverage_decay))
    wards_proj['coverage'] = wards_proj['coverage'].fillna(0.0)
    return wards_proj


def gini_coefficient(array_like):
    a = np.array(array_like, dtype=float)
    if a.size == 0:
        return np.nan
    if np.any(a < 0):
        a = a - a.min()
    if a.sum() == 0:
        return 0.0
    a_sorted = np.sort(a)
    n = a_sorted.size
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * a_sorted)) / (n * a_sorted.sum()) - (n + 1.0) / n


def plot_ward_choropleth(wards_proj, stations_list, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    wards_proj.plot(column='coverage', cmap='YlOrRd', legend=True, ax=ax,
                    legend_kwds={'label': 'Coverage score', 'shrink': 0.6}, missing_kwds={'color': 'lightgrey'})
    stations_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in stations_list], crs=wards_proj.crs)
    stations_gdf.plot(ax=ax, color='black', markersize=20, label='Stations')
    ax.set_title(f'Ward-level Coverage (Gini = {gini_coefficient(wards_proj["coverage"].values):.3f})', fontsize=14)
    ax.axis('off')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bottom_n(wards_proj, n=10, out_path=None):
    bottom = wards_proj[['coverage', 'station_count']].sort_values('coverage').head(n)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bottom['coverage'].plot(kind='bar', color='orange', ax=ax)
    ax.set_ylabel('Coverage score')
    ax.set_title(f'{n} Most Underserved Wards (by coverage)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return fig

# End of file

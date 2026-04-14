import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point


def main():
    root = Path(r"d:/Delhi Implementation")
    parquet = root / "ev_placement" / "processed_segments.parquet"
    landuse_shp = root / "landuse.shp"
    stations_geo = root / "charging_stations (1).geojson"
    city_geo = root / "new_delhi.json"
    out_dir = root / "ev_placement"
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading parquet...', parquet)
    df = pd.read_parquet(parquet)
    print('rows', len(df))

    # Convert geometry_wkt to GeoDataFrame
    df['geometry'] = df['geometry_wkt'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    print('Reprojecting segments to EPSG:32643 (metric)')
    gdf = gdf.to_crs(epsg=32643)

    # Load city polygon
    print('Loading city polygon...')
    city = gpd.read_file(root / 'new_delhi_traffic_dataset' / 'geojson' / 'new_delhi.json')
    if city.crs is None:
        city.set_crs(epsg=4326, inplace=True)
    city = city.to_crs(epsg=32643)
    bounds = city.total_bounds  # minx, miny, maxx, maxy
    minx, miny, maxx, maxy = bounds
    print('City bounds (metric):', bounds)

    # Grid parameters
    res = 100.0  # meters per pixel
    # compute grid dims
    width = int(math.ceil((maxx - minx) / res))
    height = int(math.ceil((maxy - miny) / res))
    print('Grid size (h,w):', height, width, 'resolution m:', res)

    # initialize arrays
    demand_avg = np.zeros((height, width), dtype=np.float32)
    demand_peak = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)

    # helper to convert metric x,y to pixel indices
    def xy_to_pix(x, y):
        px = int((x - minx) // res)
        py = int((maxy - y) // res)  # y-axis inverted
        if px < 0 or px >= width or py < 0 or py >= height:
            return None
        return py, px

    # rasterize segments by sampling along lines
    print('Rasterizing segments by sampling along LineStrings...')
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        length = geom.length
        # sample spacing
        step = max(10.0, res / 2.0)  # 10m or half-pixel
        n = max(1, int(math.ceil(length / step)))
        for j in range(n+1):
            try:
                pt = geom.interpolate(j / n, normalized=True)
            except Exception:
                continue
            coords = pt.coords[0]
            pix = xy_to_pix(coords[0], coords[1])
            if pix is None:
                continue
            py, px = pix
            demand_avg[py, px] += float(row.get('avg_probe_count', 0.0) or 0.0)
            demand_peak[py, px] += float(row.get('peak_probe_count', 0.0) or 0.0)
            counts[py, px] += 1

    # Optionally normalize by counts to get average per-sample
    mask = counts > 0
    demand_avg[mask] = demand_avg[mask] / counts[mask]
    demand_peak[mask] = demand_peak[mask] / counts[mask]

    # Load landuse and rasterize to R1 suitability
    print('Loading landuse shapefile...', landuse_shp)
    if landuse_shp.exists():
        lu = gpd.read_file(landuse_shp)
        if lu.crs is None:
            lu.set_crs(epsg=4326, inplace=True)
        lu = lu.to_crs(epsg=32643)
        # try to find an R1-like column
        r1_col = None
        for c in lu.columns:
            if c.lower() in ('r1_value', 'r1', 'suitability', 'suitab'):
                r1_col = c
                break
        if r1_col is None:
            # attempt to derive: treat non-residential as suitable
            # inspect likely landuse/type column
            candidate = None
            for c in lu.columns:
                if c.lower() in ('landuse', 'lu', 'type', 'class', 'landuse1'):
                    candidate = c
                    break
            if candidate is not None:
                vals = lu[candidate].astype(str)
                # simple heuristic
                suitable = vals.str.contains('Commercial|Industrial|Retail|Market|Business|Office|Hospital|Education|Transport', case=False, na=False)
                lu['R1_Value'] = suitable.astype(int)
                r1_col = 'R1_Value'
            else:
                # fallback: set all to 1
                lu['R1_Value'] = 1
                r1_col = 'R1_Value'

        # build grid point centers
        print('Building grid point centers for spatial join...')
        xs = minx + (np.arange(width) + 0.5) * res
        ys = maxy - (np.arange(height) + 0.5) * res
        pts = []
        for yi in range(height):
            y = ys[yi]
            for xi in range(width):
                x = xs[xi]
                pts.append(Point(x, y))
        pts_gdf = gpd.GeoDataFrame({'geometry': pts}, geometry='geometry', crs='EPSG:32643')
        print('Performing spatial join (this may take a moment)...')
        joined = gpd.sjoin(pts_gdf, lu[[r1_col, 'geometry']], how='left', predicate='within')
        # ensure we reindex to the original grid order because sjoin may drop rows
        # sjoin may produce multiple matches for a single grid point, so aggregate by index
        r1_series = joined[r1_col].groupby(joined.index).first()
        r1_full = r1_series.reindex(pts_gdf.index).fillna(0)
        r1_vals = r1_full.to_numpy(dtype=np.float32)
        landuse_r1 = r1_vals.reshape((height, width))
    else:
        print('Landuse shapefile not found, creating default R1 (ones)')
        landuse_r1 = np.ones((height, width), dtype=np.float32)

    # Load charging stations and rasterize to binary mask
    print('Loading charging stations...', stations_geo)
    if stations_geo.exists():
        st = gpd.read_file(stations_geo)
        if st.crs is None:
            st.set_crs(epsg=4326, inplace=True)
        st = st.to_crs(epsg=32643)
        stations_mask = np.zeros((height, width), dtype=np.uint8)
        for _, srow in st.iterrows():
            if srow.geometry is None:
                continue
            x, y = srow.geometry.x, srow.geometry.y
            pix = xy_to_pix(x, y)
            if pix is not None:
                stations_mask[pix[0], pix[1]] = 1
    else:
        print('No charging_stations file; using empty mask')
        stations_mask = np.zeros((height, width), dtype=np.uint8)

    # distance transform (requires scipy)
    try:
        from scipy import ndimage
        inv = 1 - stations_mask
        dist_pix = ndimage.distance_transform_edt(inv)
        stations_distance = dist_pix * res
    except Exception as e:
        print('scipy not available or failed distance transform:', e)
        stations_distance = np.full((height, width), np.nan, dtype=np.float32)

    # Save outputs
    print('Saving arrays...')
    np.save(out_dir / 'demand_avg.npy', demand_avg)
    np.save(out_dir / 'demand_peak.npy', demand_peak)
    np.save(out_dir / 'landuse_r1.npy', landuse_r1)
    np.save(out_dir / 'stations_mask.npy', stations_mask)
    np.save(out_dir / 'stations_distance.npy', stations_distance)

    meta = {
        'crs': 'EPSG:32643',
        'resolution_m': res,
        'width': width,
        'height': height,
        'bounds': [float(minx), float(miny), float(maxx), float(maxy)]
    }
    with open(out_dir / 'raster_meta.json', 'w', encoding='utf8') as fh:
        json.dump(meta, fh, indent=2)

    print('Done. Saved arrays to', out_dir)


if __name__ == '__main__':
    main()

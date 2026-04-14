#!/usr/bin/env python3
"""
Stream and aggregate probe_counts GeoJSONs into a Parquet dataset for modeling.

Saves per-segment aggregates: avg_probe_count, peak_probe_count, peak_timeSet, and core properties.

This script uses fiona to stream features so it can handle large GeoJSON files without reading everything into memory.
"""
import argparse
import fiona
import os
import json
from tqdm import tqdm
import pandas as pd
from shapely.geometry import shape


def process_file(path, out_rows):
    with fiona.open(path, 'r') as src:
        for feat in src:
            props = feat.get('properties', {}) or {}
            geom = feat.get('geometry')
            if geom is None:
                # skip invalid
                continue
            try:
                geom_obj = shape(geom)
                geom_wkt = geom_obj.wkt
            except Exception:
                geom_wkt = None

            # segmentProbeCounts is expected to be an array of objects with probeCount and timeSet
            seg_counts = props.get('segmentProbeCounts') or []
            # Some files may have empty arrays
            total = 0
            count_entries = 0
            peak = -1
            peak_time = None
            for sc in seg_counts:
                try:
                    pc = int(sc.get('probeCount', 0))
                except Exception:
                    pc = 0
                total += pc
                count_entries += 1
                if pc > peak:
                    peak = pc
                    peak_time = sc.get('timeSet')

            avg = (total / count_entries) if count_entries > 0 else 0

            out_rows.append({
                'segmentId': props.get('segmentId'),
                'newSegmentId': props.get('newSegmentId'),
                'avg_probe_count': avg,
                'peak_probe_count': peak if peak >= 0 else 0,
                'peak_timeSet': peak_time,
                'speedLimit': props.get('speedLimit'),
                'frc': props.get('frc'),
                'streetName': props.get('streetName'),
                'distance': props.get('distance'),
                'geometry_wkt': geom_wkt,
                'source_file': os.path.basename(path)
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with probe_counts GeoJSON files')
    parser.add_argument('--output', required=True, help='Output Parquet file path')
    args = parser.parse_args()

    files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.geojson')])
    if not files:
        print('No geojson files found in', args.input_dir)
        return

    rows = []
    for p in tqdm(files, desc='Files'):
        try:
            process_file(p, rows)
        except Exception as e:
            print('Error processing', p, e)

    df = pd.DataFrame(rows)
    # Basic dedupe: keep highest avg per newSegmentId
    if 'newSegmentId' in df.columns:
        df = df.sort_values(['newSegmentId','avg_probe_count'], ascending=[True,False]).drop_duplicates('newSegmentId')

    # Save to parquet
    out_path = args.output
    df.to_parquet(out_path, index=False)
    print('Saved', len(df), 'rows to', out_path)


if __name__ == '__main__':
    main()

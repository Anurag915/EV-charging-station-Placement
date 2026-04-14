Data preprocessing for EV charging placement project

This folder contains a small preprocessing script to convert the probe_counts GeoJSON files into a single, cleaned Parquet file with per-segment aggregated metrics suitable for modeling and visualization.

Usage (from repository root):

1. Create and activate a Python environment (recommended Python 3.10+).
2. Install requirements: pip install -r ev_placement/requirements.txt
3. Run the preprocessing:
   python ev_placement/data_preprocess.py \
       --input_dir "d:/Delhi Implementation/new_delhi_traffic_dataset/probe_counts/geojson" \
       --output "d:/Delhi Implementation/ev_placement/processed_segments.parquet"

Output: a Parquet file with columns: segmentId, newSegmentId, avg_probe_count, peak_probe_count, peak_timeSet, speedLimit, frc, streetName, distance, geometry_wkt

This is a first-step artifact to prepare the dataset for the A2C training pipeline and interactive visualization.

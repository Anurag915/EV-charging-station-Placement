# 📊 Dataset Requirements

To minimize the repository size and comply with GitHub's 100MB file limit, several large raw datasets have been excluded from the version control system. 

While the project includes **pre-processed numpy layers** in the `ev_placement/` directory (enough to run the dashboard and evaluation scripts), if you want to perform full data pre-processing or retraining, you will need the following datasets.

---

## 📂 Required Directories
The following folders are ignored by Git but are expected by the pre-processing scripts:

### 1. `new_delhi_traffic_dataset/`
- **Purpose**: Provides high-resolution probe counts for traffic demand modeling.
- **Expected Files**: 
  - `probe_counts/geojson/new_delhi__2024-08-11_to_2024-08-11_.geojson` (~116MB)
  - Other date-specific GeoJSON files.
- **Source**: Derived from traffic sensor data or open-source mobility datasets.

### 2. `LandUseDataset/` & `Land_use_mrg/`
- **Purpose**: GIS data defining the zoning of Delhi (Industrial, Residential, Commercial, Green Belt).
- **Expected Files**:
  - `landuse.shp`, `landuse.shx`, `landuse.dbf`
  - `Merged_Delhi.shp` (~2MB)
- **Source**: [MapCruzin.com](https://www.mapcruzin.com) or OpenStreetMap exports.

---

## 🏗️ Data Structure
If you are adding these datasets back to the folder, ensure the structure looks like this:

```text
/
├── new_delhi_traffic_dataset/
│   └── probe_counts/
│       └── geojson/
│           └── new_delhi__2024-08-11_to_2024-08-11_.geojson
├── LandUseDataset/
│   └── (shapefile components)
└── Land_use_mrg/
    └── (merged shapefile components)
```

---

## ⚙️ How to Regenerate Pre-processed Layers
If you have the raw datasets and want to regenerate the `.npy` files used by the model:

1.  Navigate to the `ev_placement` core folder.
2.  Run the rasterization scripts:
    ```python
    python ev_placement/rasterize_channels.py
    python ev_placement/data_preprocess.py
    ```

---
*Note: The core project logic remains fully functional without these if you use the pre-baked results provided in this repo.*

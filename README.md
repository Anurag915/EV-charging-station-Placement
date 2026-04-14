# ⚡ Delhi EV Charging Station Placement Optimization

A Deep Reinforcement Learning (DRL) project designed to optimize Electric Vehicle (EV) charging station placement across Delhi. Using an Enhanced Time-Aware A2C (Advantage Actor-Critic) model, this system adapts to complex urban dynamics, including rush-hour traffic cycles and diverse land-use patterns.

---

## 🌟 Key Features

- **Temporal Awareness**: An LSTM-based architecture that remembers daily traffic cycles (Morning/Evening Rush Hour).
- **Multi-Modal Spatial Input**: Integrates Land Use (Green Belts, Water Bodies, Residential, Commercial) and Traffic Demand layers.
- **Enhanced Reward Logic**: 
  - Rewards placements in high-traffic commercial zones.
  - Penalizes clustering (cluttering) of stations to ensure optimal coverage.
  - Avoids restricted areas like water bodies and protected green belts.
- **Interactive Dashboard**: A Streamlit interface with Folium maps to compare manual placements against model-proposed candidates.
- **Evaluation Framework**: Automated comparisons between static baseline agents and the improved temporal-aware agent.

---

## 📂 Repository Structure

- `dashboard.py`: Interactive Streamlit dashboard for geo-visualization.
- `improved_temporal_training.py`: Core training script for the time-aware A2C agent.
- `evaluate_baseline_vs_improved.py`: Performance benchmarking and metrics generation.
- `ev_placement/`: Pre-processed spatial data layers (compressed numpy arrays).
- `temporal_results/`: Stores trained model weights and training logs.
- `DATASET_REQUIREMENTS.md`: Guidance on the large traffic and GIS datasets required to retrain the model.

---

## 🛠️ Setup and Installation

### 1. Prerequisite Environments
Ensure you have Python 3.8+ installed.

### 2. Clone and Install
```bash
git clone https://github.com/Anurag915/EV-charging-station-Placemnt.git
cd EV-charging-station-Placemnt
pip install -r requirements.txt
```

### 3. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env
```
*(Currently no external API keys are required for basic usage, but this is a placeholder for future Mapbox/Google Maps integrations).*

---

## 🚀 Running the Project

### Interactive Dashboard
Visualize current placements and model suggestions:
```bash
streamlit run dashboard.py
```

### Performance Demonstration
Run a simulation of temporal behavior across various time steps:
```bash
python demonstrate_temporal_features.py
```

### Full Evaluation
Compare the baseline model vs. the improved version:
```bash
python evaluate_baseline_vs_improved.py
```

---

## 📊 Dataset Notice
To keep the repository lightweight, raw GIS datasets (>100MB) are excluded. If you wish to retrain the model or update the spatial layers, please refer to [DATASET_REQUIREMENTS.md](DATASET_REQUIREMENTS.md).

---

## 📜 Acknowledgments
- **MapCruzin**: For providing the core Delhi GIS shapefiles.
- **OpenStreetMap**: Source data for road networks and infrastructure.
- **Delhi Master Plan 2041**: Rationale for land-use priority and placement constraints.

---
*Created for optimization of urban EV infrastructure.*

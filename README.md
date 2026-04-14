# Delhi EV Charging Station Placement Optimization

This project implements a **Deep Reinforcement Learning (DRL)** approach to optimize the placement of Electric Vehicle (EV) charging stations in Delhi. It uses an **Enhanced Time-Aware A2C (Advantage Actor-Critic)** agent that incorporates **Temporal LSTM** networks to adapt to rush-hour and off-peak traffic patterns.

## 🚀 Key Features

- **Temporal Awareness**: Adapts placement strategies based on time-of-day (Rush Hour vs. Off-Peak).
- **Multi-Modal Input**: Combines spatial data (Land Use, Existing Infrastructure, Traffic) with temporal data.
- **Advanced Architecture**: Uses spatial CNNs, Temporal LSTMs, and Attention mechanisms (6.4M parameters).
- **Interactive Dashboard**: Streamlit dashboard for visualizing results and performance.

## 📂 Project Structure

- `finalyearproject.ipynb`: Main implementation notebook.
- `demonstrate_temporal_features.py`: Script to demonstrate temporal features and generate plots.
- `dashboard.py`: Interactive Streamlit dashboard.
- `improved_temporal_training.py`: Training script with improved positive reward function.
- `ev_placement/`: Directory containing processed data layers (Land Use, Demand, etc.).
- `temporal_results/`: Directory storing trained models and results.

## 🛠️ Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Dashboard**:
    ```bash
    streamlit run dashboard.py
    ```

3.  **Run Temporal Demonstration**:
    ```bash
    python demonstrate_temporal_features.py
    ```

## 📊 Results

The temporal model achieves a **+19.9% improvement** (with improved positive rewards) over the baseline static A2C approach, effectively optimizing for:
- **Morning Rush (7-10 AM)**
- **Evening Rush (5-8 PM)**
- **Off-Peak Efficiency (10 PM - 6 AM)**

For more details, refer to the `Final_Report.pdf` or `Presentation_Content.md`.

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from reward_calculator import RewardCalculator

st.set_page_config(layout="wide", page_title="Delhi 2041 Master Plan EV Planner")

st.title("⚡ Delhi 2041 EV Planner (Image Overlaid)")
st.markdown("""
**Data Sources:**
*   **Land Use**: Extracted directly from uploaded map screenshot (Real Overlay).
*   **Traffic**: Modeled on Delhi Road Network (Ring Roads & Arterials).
""")

if 'calculator' not in st.session_state:
    st.session_state.calculator = RewardCalculator()
if 'placed_stations' not in st.session_state:
    st.session_state.placed_stations = []

calc = st.session_state.calculator

# Sidebar
st.sidebar.header("Network Simulator")
st.sidebar.metric("Stations Placed", len(st.session_state.placed_stations))
if st.sidebar.button("Reset Network"):
    st.session_state.placed_stations = []
    st.rerun()

layer = st.sidebar.radio("Map Layer", ["None", "Road Network Traffic", "Real Land Use (Image)"])

col_map, col_details = st.columns([2, 1])

with col_map:
    m = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
    
    # 1. Existing Infra
    for lat, lon in calc.existing_stations:
        folium.CircleMarker(
            [lat, lon], radius=3, color='red', fill=True, tooltip="Existing Station"
        ).add_to(m)
        
    # 2. Placed (Manual)
    for lat, lon in st.session_state.placed_stations:
        folium.Marker([lat, lon], icon=folium.Icon(color='green', icon='bolt', prefix='fa'), tooltip="Manual Placement").add_to(m)

    # 3. Model Proposed (Projected)
    show_model = st.sidebar.checkbox("Show Model Proposed Stations", value=True)
    if show_model:
        import os
        import json
        model_file = "model_placements.json"
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                    for pt in model_data:
                        # Assuming format is list of [lat, lon] or similar
                        # flexible handling
                        if isinstance(pt, dict): lat, lon = pt['lat'], pt['lon']
                        else: lat, lon = pt[0], pt[1]
                        
                        folium.Marker(
                            [lat, lon], 
                            icon=folium.Icon(color='orange', icon='star', prefix='fa'),
                            tooltip=f"Model Proposed (Reward: {pt.get('reward', 'N/A') if isinstance(pt, dict) else 'N/A'})"
                        ).add_to(m)
            except Exception as e:
                st.sidebar.error(f"Error loading model placements: {e}")

    # 3. Layers
    if layer == "Road Network Traffic":
        # Visualize the Web
        rows, cols = np.where(calc.traffic > 0.4)
        if len(rows) > 1000:
             idx = np.random.choice(len(rows), 1000, replace=False)
             rows, cols = rows[idx], cols[idx]
             
        for r, c in zip(rows, cols):
            val = calc.traffic[r, c]
            lon, lat = calc.grid_to_geo(r, c)
            dlat = (calc.maxy - calc.miny)/50
            dlon = (calc.maxx - calc.minx)/50
            folium.Rectangle(
                bounds=[[lat-dlat/2, lon-dlon/2], [lat+dlat/2, lon+dlon/2]],
                color="red", weight=0, fill=True, fill_opacity=val*0.6,
                tooltip="High Traffic Road"
            ).add_to(m)
            
    elif layer == "Real Land Use (Image)":
        color_map = {
            0: '#006400', # Green Belt
            1: '#1E90FF', # Water
            2: '#FFFF00', # Res (Yellow)
            3: '#FF0000', # Comm (Red)
            5: '#800080', # Ind (Purple)
        }
        
        for code, color in color_map.items():
            if code == 2: continue # Skip yellow base
            
            rows, cols = np.where(calc.land_use == code)
            
            # Subsample for perf
            if len(rows) > 800:
                idx = np.random.choice(len(rows), 800, replace=False)
                rows, cols = rows[idx], cols[idx]
                
            for r, c in zip(rows, cols):
                lon, lat = calc.grid_to_geo(r, c)
                dlat = (calc.maxy - calc.miny)/50
                dlon = (calc.maxx - calc.minx)/50
                
                folium.Rectangle(
                    bounds=[[lat-dlat/2, lon-dlon/2], [lat+dlat/2, lon+dlon/2]],
                    color=color, weight=0, fill=True, fill_opacity=0.4,
                    tooltip=f"Zone: {code}"
                ).add_to(m)

    output = st_folium(m, width=800, height=500)

with col_details:
    st.subheader("Site Analysis")
    if output['last_clicked']:
        lat = output['last_clicked']['lat']
        lon = output['last_clicked']['lng']
        
        res = calc.assess_location(lat, lon, st.session_state.placed_stations)
        
        st.metric("Suitability Score", f"{res['Total']:.1f}")
        
        st.write(f"**Description**: {res['Land_Reason']}")
        
        if res['R_Land'] > 15: st.success("Target Zone (High Priority)")
        elif res['R_Land'] < 0: st.error("Restricted/Penalty Zone")
        else: st.info("Standard Zone")
        
        st.divider()
        st.write(f"**Traffic**: {res['R_Traffic']}")
        st.write(f"**Crowding**: {res['R_Crowding']}")
        
        is_dup = any(abs(stat[0]-lat)<1e-5 and abs(stat[1]-lon)<1e-5 for stat in st.session_state.placed_stations)
        if not is_dup:
            if st.button("📍 Place Station", type="primary"):
                st.session_state.placed_stations.append((lat, lon))
                st.rerun()
    else:
        st.info("Select a location on the map.")

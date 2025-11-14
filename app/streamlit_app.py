import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Beautiful Pastel Color Palette
COLORS = {
    'primary': '#6C9BCF', 'secondary': '#9DCCFF', 'accent': '#FFB4A8',
    'success': '#B5EAD7', 'warning': '#FFE8A8', 'danger': '#FFB4C8',
    'info': '#C7CEEA', 'purple': '#E7C6FF', 'teal': '#A8DADC'
}

# Page config
st.set_page_config(page_title="Koramangala Environmental Monitor", page_icon="🌍", layout="wide")

# Custom CSS
st.markdown(f"""
<style>
.stMetric {{background:{COLORS['teal']}20; padding: 15px; border-radius: 10px; border-left: 4px solid {COLORS['primary']};}}
h1 {{color:{COLORS['primary']}; font-weight: 600;}}
</style>
""", unsafe_allow_html=True)

#=== GNN+LSTM TRAFFIC PREDICTION MODEL ===
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj_matrix):
        x = torch.matmul(adj_matrix, x)
        return torch.relu(self.linear(x))

class GNN_LSTM_TrafficModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_nodes=10, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.gnn1 = GraphConvLayer(input_dim, hidden_dim)
        self.gnn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adj_matrix):
        x = self.gnn1(x, adj_matrix)
        x = self.gnn2(x, adj_matrix)
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.fc(x.squeeze(0))
        return x

#=== SCHOOL DATA ===
KORAMANGALA_SCHOOLS = [
    {"name": "Delhi Public School", "lat": 12.9343, "lon": 77.6210, "start": "7:30", "end": "15:00"},
    {"name": "National Public School", "lat": 12.9280, "lon": 77.6270, "start": "8:00", "end": "15:30"},
    {"name": "Indus International School", "lat": 12.9410, "lon": 77.6310, "start": "8:15", "end": "15:45"},
    {"name": "Inventure Academy", "lat": 12.9290, "lon": 77.6180, "start": "7:45", "end": "15:15"},
    {"name": "Greenwood High", "lat": 12.9320, "lon": 77.6240, "start": "8:00", "end": "15:30"}
]

def is_school_hours():
    now = datetime.now()
    current_time = now.time()
    for school in KORAMANGALA_SCHOOLS:
        start = datetime.strptime(school["start"], "%H:%M").time()
        end = datetime.strptime(school["end"], "%H:%M").time()
        if start <= current_time <= end or (now.hour in [7,8,15,16]):
            return True
    return False

def get_traffic_multiplier():
    now = datetime.now()
    hour = now.hour
    day = now.weekday()
    
    if day >= 5:
        return 0.6
    
    if hour in [7, 8] or hour in [15, 16]:
        return 1.8
    elif hour in [9, 10]:
        return 1.5
    elif hour in [17, 18, 19]:
        return 1.6
    elif hour in [12, 13]:
        return 1.3
    elif hour >= 23 or hour <= 5:
        return 0.3
    else:
        return 1.0

def generate_dynamic_traffic():
    base_traffic = 45
    multiplier = get_traffic_multiplier()
    noise = np.random.normal(0, 5)
    traffic = base_traffic * multiplier + noise
    return max(0, min(100, traffic))

def predict_traffic_gnn_lstm():
    model = GNN_LSTM_TrafficModel()
    num_nodes = 10
    adj_matrix = torch.eye(num_nodes) + torch.rand(num_nodes, num_nodes) * 0.3
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    x = torch.randn(num_nodes, 5) * 0.5 + torch.tensor([[get_traffic_multiplier()] * 5])
    model.eval()
    with torch.no_grad():
        predictions = model(x, adj_matrix)
    return predictions.numpy().flatten()

def generate_traffic_forecast(hours=6):
    current = generate_dynamic_traffic()
    forecast = [current]
    for i in range(1, hours):
        future_hour = (datetime.now() + timedelta(hours=i)).hour
        base = 45
        if future_hour in [7, 8, 15, 16]:
            multiplier = 1.8
        elif future_hour in [9, 10]:
            multiplier = 1.5
        elif future_hour in [17, 18, 19]:
            multiplier = 1.6
        else:
            multiplier = 1.0
        forecast.append(base * multiplier + np.random.normal(0, 3))
    return forecast

# Header
st.markdown(f"""
<div style='background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']}); padding: 20px; border-radius: 15px;'>
<h1 style='color: white; text-align: center; margin: 0;'>🌍 Koramangala Environmental Monitoring</h1>
<p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>Real-time Air Quality | Traffic | ML Predictions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    view = st.selectbox("View", ["📊 Real-time", "📈 Historical", "🔮 Predictions"])
    time_range = st.slider("Hours", 1, 168, 24)
    st.markdown("---")
    st.info("**Area:** Koramangala, Bengaluru\n\n**Coords:** 12.9352° N, 77.6245° E")

# Generate data
np.random.seed(42)
current_time = datetime.now()
current_aqi = np.random.randint(120, 180)
current_temp = round(np.random.uniform(24, 32), 1)
current_pm25 = np.random.randint(50, 120)
current_traffic = generate_dynamic_traffic()
school_hours = is_school_hours()

# School hours alert
if school_hours:
    st.warning("🏫 **School Hours Active**: Traffic congestion expected near schools. Multiplier: 1.8x")

# Metrics
st.markdown("## 📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🌫️ AQI", current_aqi, "+12", delta_color="inverse")
with col2:
    st.metric("🌡️ Temp", f"{current_temp}°C", "+1.2°C")
with col3:
    st.metric("💨 PM2.5", current_pm25, "+8", delta_color="inverse")
with col4:
    st.metric("🚗 Traffic", f"{current_traffic:.0f}", f"x{get_traffic_multiplier():.1f}")
with col5:
    st.metric("⚠️ Alerts", 3, "+1", delta_color="inverse")

st.markdown("---")

# Map
st.markdown("## 🗺️ Air Quality & Traffic Map")
m = folium.Map([12.9352, 77.6245], zoom_start=14, tiles='CartoDB positron')

# Add AQ

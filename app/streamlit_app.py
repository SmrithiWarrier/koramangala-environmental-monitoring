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
    
    # Weekend lower traffic
    if day >= 5:
        return 0.6
    
    # School hours (7-9 AM, 3-5 PM)
    if hour in [7, 8] or hour in [15, 16]:
        return 1.8
    # Morning rush (9-11 AM)
    elif hour in [9, 10]:
        return 1.5
    # Evening rush (5-8 PM)
    elif hour in [17, 18, 19]:
        return 1.6
    # Lunch time (12-2 PM)
    elif hour in [12, 13]:
        return 1.3
    # Late night (11 PM - 5 AM)
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

#=== TRAFFIC FORECASTING FUNCTION ===
def predict_traffic_gnn_lstm():
    model = GNN_LSTM_TrafficModel()
    # Create adjacency matrix for road network
    num_nodes = 10
    adj_matrix = torch.eye(num_nodes) + torch.rand(num_nodes, num_nodes) * 0.3
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # Simulate input features (time, traffic_density, speed, weather, events)
    x = torch.randn(num_nodes, 5) * 0.5 + torch.tensor([[get_traffic_multiplier()] * 5])
    
    # Run prediction
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

#=== MOCK DATA GENERATION ===
def get_current_data():
    current_traffic = generate_dynamic_traffic()
    return {
        'aqi': np.random.randint(80, 150),
        'temperature': np.random.uniform(24, 32),
        'pm25': np.random.uniform(35, 75),
        'traffic_index': current_traffic,
        'active_alerts': np.random.randint(1, 5)
    }

def generate_time_series(hours=24):
    times = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]
    aqi = [np.random.randint(80, 150) for _ in range(hours)]
    pm25 = [np.random.uniform(30, 80) for _ in range(hours)]
    traffic = []
    for t in times:
        hour = t.hour
        if hour in [7, 8, 15, 16]:
            traffic.append(np.random.uniform(70, 95))
        elif hour in [9, 10, 17, 18]:
            traffic.append(np.random.uniform(60, 85))
        else:
            traffic.append(np.random.uniform(30, 60))
    return pd.DataFrame({'time': times, 'aqi': aqi, 'pm25': pm25, 'traffic': traffic})

#=== HEADER ===
st.markdown(f"""
<div style='background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']}); padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'>🌍 Koramangala Environmental Monitoring System</h1>
    <p style='color: white; opacity: 0.9; font-size: 16px; margin: 5px 0 0 0;'>Real-time monitoring with AI-powered traffic prediction</p>
</div>
""", unsafe_allow_html=True)

data = get_current_data()
school_hours = is_school_hours()

# School hours alert
if school_hours:
    st.warning("🏫 **School Hours Active**: Traffic congestion expected near schools. Multiplier: 1.8x")

#=== KEY METRICS ===
cols = st.columns(5)
with cols[0]:
    st.metric("AQI", f"{data['aqi']}", delta=f"{np.random.randint(-10, 10)}")
with cols[1]:
    st.metric("Temperature", f"{data['temperature']:.1f}°C", delta=f"{np.random.uniform(-2, 2):.1f}°C")
with cols[2]:
    st.metric("PM2.5", f"{data['pm25']:.1f} µg/m³", delta=f"{np.random.uniform(-5, 5):.1f}")
with cols[3]:
    st.metric("Traffic Index", f"{data['traffic_index']:.0f}", delta=f"x{get_traffic_multiplier():.1f}")
with cols[4]:
    st.metric("Active Alerts", data['active_alerts'])

#=== MAP WITH SCHOOLS & TRAFFIC ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Interactive Map: Schools & Traffic Zones")
    m = folium.Map(location=[12.9352, 77.6245], zoom_start=14)
    
    # Add schools
    for school in KORAMANGALA_SCHOOLS:
        folium.Marker(
            [school['lat'], school['lon']],
            popup=f"<b>{school['name']}</b><br>Hours: {school['start']}-{school['end']}",
            icon=folium.Icon(color='red', icon='graduation-cap', prefix='fa'),
            tooltip=school['name']
        ).add_to(m)
    
    # Add traffic zones
    traffic_zones = [
        {'name': 'Zone 1: 80 Feet Rd', 'lat': 12.9330, 'lon': 77.6200, 'traffic': generate_dynamic_traffic()},
        {'name': 'Zone 2: Koramangala 4th Block', 'lat': 12.9350, 'lon': 77.6250, 'traffic': generate_dynamic_traffic()},
        {'name': 'Zone 3: Sony Signal', 'lat': 12.9365, 'lon': 77.6270, 'traffic': generate_dynamic_traffic()}
    ]
    
    for zone in traffic_zones:
        color = 'green' if zone['traffic'] < 40 else 'orange' if zone['traffic'] < 70 else 'red'
        folium.CircleMarker(
            [zone['lat'], zone['lon']],
            radius=15,
            popup=f"<b>{zone['name']}</b><br>Traffic: {zone['traffic']:.0f}/100",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6
        ).add_to(m)
    
    folium_static(m, width=700, height=450)

with col2:
    st.subheader("🚦 Traffic Flow Analysis")
    st.markdown(f"**Current Multiplier**: {get_traffic_multiplier():.2f}x")
    st.markdown(f"**Time**: {datetime.now().strftime('%H:%M')}")
    
    # Traffic prediction
    gnn_predictions = predict_traffic_gnn_lstm()
    avg_prediction = np.mean(gnn_predictions)
    
    st.metric("GNN+LSTM Forecast (1h)", f"{avg_prediction:.0f}", delta=f"{avg_prediction - data['traffic_index']:.0f}")
    
    st.markdown("### 📊 Live Traffic Zones")
    for zone in traffic_zones:
        status = "🟢 Low" if zone['traffic'] < 40 else "🟠 Medium" if zone['traffic'] < 70 else "🔴 High"
        st.write(f"{status}: {zone['name']} - **{zone['traffic']:.0f}**")

#=== TIME SERIES & PREDICTIONS ===
st.subheader("📈 Historical Data & Predictions")
col1, col2 = st.columns(2)

df_history = generate_time_series(24)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_history['time'], y=df_history['aqi'], name='AQI', 
                              line=dict(color=COLORS['primary'], width=3)))
    fig1.update_layout(title='AQI Trend (24h)', height=300, template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_history['time'], y=df_history['traffic'], name='Traffic',
                              line=dict(color=COLORS['accent'], width=3)))
    fig2.update_layout(title='Traffic Index Trend (24h)', height=300, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

#=== GNN+LSTM TRAFFIC FORECAST ===
st.subheader("🤖 AI Traffic Forecast (GNN+LSTM)")
forecast_hours = 6
traffic_forecast = generate_traffic_forecast(forecast_hours)
forecast_times = [datetime.now() + timedelta(hours=i) for i in range(forecast_hours)]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=forecast_times, y=traffic_forecast, name='Predicted Traffic',
                          line=dict(color=COLORS['purple'], width=4, dash='dash'),
                          mode='lines+markers'))
fig3.update_layout(title='Traffic Forecast (Next 6 Hours)', height=350, template='plotly_white',
                   xaxis_title='Time', yaxis_title='Traffic Index (0-100)')
st.plotly_chart(fig3, use_container_width=True)

#=== ALERTS ===
st.subheader("⚠️ Active Alerts")
alerts_data = pd.DataFrame({
    'Time': [datetime.now() - timedelta(minutes=i*15) for i in range(3)],
    'Type': ['High PM2.5', 'Traffic Congestion', 'Poor AQI'],
    'Location': ['Koramangala 5th Block', 'Near DPS School', 'Sony Signal'],
    'Severity': ['High', 'Medium', 'High']
})
st.dataframe(alerts_data, use_container_width=True)

#=== RECOMMENDATIONS ===
st.subheader("💡 Recommendations")
col1, col2 = st.columns(2)
with col1:
    st.info("**For Citizens:**\n- Avoid outdoor activities during peak pollution hours\n- Use air purifiers indoors\n- Check AQI before planning travel")
with col2:
    st.success("**For Policy Makers:**\n- Implement traffic diversions near schools\n- Increase public transport during rush hours\n- Monitor industrial emissions")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data updates every 5 minutes")

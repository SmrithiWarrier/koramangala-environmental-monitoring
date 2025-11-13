import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static

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
.stMetric {{background: {COLORS['teal']}20; padding: 15px; border-radius: 10px; border-left: 4px solid {COLORS['primary']};}}
h1 {{color: {COLORS['primary']}; font-weight: 600;}}
</style>
""", unsafe_allow_html=True)

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
    st.metric("🚗 Traffic", 78, "-5%")
with col5:
    st.metric("⚠️ Alerts", 3, "+1", delta_color="inverse")

st.markdown("---")

# Map
st.markdown("## 🗺️ Air Quality Map")
m = folium.Map([12.9352, 77.6245], zoom_start=14, tiles='CartoDB positron')
points = [
    {"coords": [12.9352, 77.6245], "name": "Central", "aqi": 156},
    {"coords": [12.9412, 77.6281], "name": "North", "aqi": 142},
    {"coords": [12.9292, 77.6209], "name": "South", "aqi": 178},
    {"coords": [12.9365, 77.6350], "name": "East", "aqi": 165},
    {"coords": [12.9339, 77.6140], "name": "West", "aqi": 138}
]
for p in points:
    color = "red" if p["aqi"] > 150 else "orange" if p["aqi"] > 100 else "green"
    folium.CircleMarker(p["coords"], radius=15, popup=f"{p['name']}: {p['aqi']}", 
                       color=color, fill=True, fillColor=color, fillOpacity=0.7).add_to(m)
folium_static(m, width=1200, height=500)

st.markdown("---")

# Charts
st.markdown("## 📈 Temporal Analysis")
dates = pd.date_range(end=current_time, periods=time_range, freq='H')
aqi_data = 150 + np.random.randn(time_range).cumsum() * 3

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=aqi_data, mode='lines', name='AQI',
                            line=dict(color=COLORS['primary'], width=3), fill='tozeroy'))
    fig.add_hline(y=150, line_dash="dash", line_color=COLORS['danger'], annotation_text="Unhealthy")
    fig.update_layout(title="🌫️ AQI Trend", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pm25 = 75 + np.random.randn(time_range).cumsum() * 2
    pm10 = 110 + np.random.randn(time_range).cumsum() * 3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=pm25, name='PM2.5', line=dict(color=COLORS['accent'])))
    fig.add_trace(go.Scatter(x=dates, y=pm10, name='PM10', line=dict(color=COLORS['success'])))
    fig.update_layout(title="💨 PM Levels", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Predictions
st.markdown("## 🔮 Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Next 6 Hours")
    pred = pd.DataFrame({'Hour': [f'+{i}h' for i in range(1,7)], 
                        'AQI': np.random.randint(140, 180, 6)})
    st.dataframe(pred, use_container_width=True, hide_index=True)

with col2:
    st.subheader("24h Forecast")
    fig = px.bar(x=[f'+{i}h' for i in range(24)], y=np.random.randint(100, 200, 24))
    fig.update_traces(marker_color=COLORS['secondary'])
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Weekly")
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig = go.Figure(data=[go.Bar(x=days, y=np.random.randint(120,180,7), marker_color=COLORS['accent'])])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Alerts
st.markdown("## ⚠️ Active Alerts")
alerts = pd.DataFrame({
    'Time': [(current_time - timedelta(hours=i)).strftime('%H:%M') for i in range(1,4)],
    'Type': ['High PM2.5', 'Traffic Jam', 'Anomaly'],
    'Severity': ['🔴 High', '🟡 Medium', '🔴 Critical'],
    'Location': ['5th Block', 'Hosur Rd', 'Sarjapur Rd']
})
st.dataframe(alerts, use_container_width=True, hide_index=True)

st.markdown("---")

# Recommendations
st.markdown("## 💡 Recommendations")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style='background:{COLORS['info']}40; padding:20px; border-radius:10px;'>
    <h3>👥 For Citizens</h3>
    <p>• Avoid outdoor activities 8-10 AM, 6-8 PM</p>
    <p>• Use N95 masks outdoors</p>
    <p>• Keep windows closed</p>
    <p>• Use air purifiers indoors</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background:{COLORS['warning']}40; padding:20px; border-radius:10px;'>
    <h3>🏛️ For Policymakers</h3>
    <p>• Implement odd-even vehicle scheme</p>
    <p>• Increase public transport</p>
    <p>• Deploy air purification units</p>
    <p>• Issue health advisory</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:{COLORS['primary']};'>
<p><b>🌍 Koramangala Environmental Monitoring System</b></p>
<p>Made with ❤️ for cleaner Koramangala | Updated every 15 min</p>
</div>
""", unsafe_allow_html=True)

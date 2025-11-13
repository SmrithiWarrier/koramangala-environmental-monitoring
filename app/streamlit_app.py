import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static

# Page configuration
st.set_page_config(
    page_title="Koramangala Environmental Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("🌍 Koramangala Environmental Monitoring System")
st.markdown("**Real-time air quality monitoring and prediction for Koramangala, Bengaluru**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    view_mode = st.selectbox("View Mode", ["Real-time", "Historical", "Predictions"])
    time_range = st.slider("Time Range (hours)", 1, 168, 24)
    st.markdown("---")
    st.markdown("📍 **Location**: Koramangala, Bengaluru")
    st.markdown("📍 **Coordinates**: 12.9352° N, 77.6245° E")
    st.markdown("---")
    st.info("This is a demo with simulated data. Connect real API keys in .env file for live data.")

# Generate demo data
np.random.seed(42)
current_time = datetime.now()

# Current metrics (simulated)
current_aqi = np.random.randint(120, 180)
aqi_change = np.random.randint(-15, 25)
current_temp = round(np.random.uniform(24, 32), 1)
temp_change = round(np.random.uniform(-2, 3), 1)
active_alerts = np.random.randint(1, 5)
alert_change = np.random.randint(-1, 2)
current_pm25 = np.random.randint(50, 120)
pm25_change = np.random.randint(-10, 20)

# Display key metrics
st.header("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Air Quality Index (AQI)",
        value=current_aqi,
        delta=f"{aqi_change:+d}",
        delta_color="inverse"
    )
    if current_aqi > 150:
        st.error("⚠️ Unhealthy")
    elif current_aqi > 100:
        st.warning("⚠️ Moderate")
    else:
        st.success("✔️ Good")

with col2:
    st.metric(
        label="Temperature (°C)",
        value=f"{current_temp}°C",
        delta=f"{temp_change:+.1f}°C"
    )

with col3:
    st.metric(
        label="Active Alerts",
        value=active_alerts,
        delta=f"{alert_change:+d}",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="PM2.5 (µg/m³)",
        value=current_pm25,
        delta=f"{pm25_change:+d}",
        delta_color="inverse"
    )

st.markdown("---")

# Map section
st.header("🗺️ Real-time Air Quality Map")

# Create Folium map
m = folium.Map(
    location=[12.9352, 77.6245],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Add monitoring points
monitoring_points = [
    {"coords": [12.9352, 77.6245], "name": "Central Koramangala", "aqi": 156},
    {"coords": [12.9412, 77.6281], "name": "North Zone", "aqi": 142},
    {"coords": [12.9292, 77.6209], "name": "South Zone", "aqi": 178},
    {"coords": [12.9365, 77.6350], "name": "East Zone", "aqi": 165},
    {"coords": [12.9339, 77.6140], "name": "West Zone", "aqi": 138},
]

for point in monitoring_points:
    # Color based on AQI
    if point["aqi"] > 150:
        color = "red"
    elif point["aqi"] > 100:
        color = "orange"
    else:
        color = "green"
    
    folium.CircleMarker(
        location=point["coords"],
        radius=15,
        popup=f"<b>{point['name']}</b><br>AQI: {point['aqi']}",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.6
    ).add_to(m)

folium_static(m, width=1200, height=500)

st.markdown("---")

# Time series charts
st.header("📊 Temporal Analysis")

# Generate time series data
dates = pd.date_range(end=current_time, periods=time_range, freq='H')
aqi_data = 150 + np.random.randn(time_range).cumsum() * 3
pm25_data = 75 + np.random.randn(time_range).cumsum() * 2
pm10_data = 110 + np.random.randn(time_range).cumsum() * 3

col1, col2 = st.columns(2)

with col1:
    # AQI time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=aqi_data,
        mode='lines',
        name='AQI',
        line=dict(color='rgb(255, 127, 14)', width=2),
        fill='tonexty'
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate Threshold", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="red", 
                  annotation_text="Unhealthy Threshold", annotation_position="right")
    
    fig.update_layout(
        title="Air Quality Index Trend",
        xaxis_title="Time",
        yaxis_title="AQI",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # PM2.5 and PM10
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=pm25_data, name='PM2.5', 
                            line=dict(color='rgb(31, 119, 180)')))
    fig.add_trace(go.Scatter(x=dates, y=pm10_data, name='PM10', 
                            line=dict(color='rgb(44, 160, 44)')))
    
    fig.update_layout(
        title="Particulate Matter Levels",
        xaxis_title="Time",
        yaxis_title="µg/m³",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Predictions section
st.header("🔮 Future Predictions")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Next 6 Hours")
    predictions_6h = pd.DataFrame({
        'Hour': [f'+{i}h' for i in range(1, 7)],
        'AQI': np.random.randint(140, 180, 6),
        'Confidence': [f"{c:.1%}" for c in np.random.uniform(0.85, 0.95, 6)]
    })
    st.dataframe(predictions_6h, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Next 24 Hours")
    future_hours = [f'+{i}h' for i in range(24)]
    future_aqi = np.random.randint(100, 200, 24)
    fig = px.bar(x=future_hours, y=future_aqi, 
                 labels={'x': 'Time', 'y': 'Predicted AQI'})
    fig.update_traces(marker_color='lightblue')
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Weekly Forecast")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_aqi = np.random.randint(120, 180, 7)
    fig = go.Figure(data=[
        go.Bar(x=days, y=weekly_aqi, marker_color='lightsalmon')
    ])
    fig.update_layout(height=300, yaxis_title="Predicted AQI")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Alerts section
st.header("⚠️ Active Alerts")

alerts_data = [
    {
        'Time': (current_time - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M'),
        'Type': 'High PM2.5 Detected',
        'Severity': 'High',
        'Location': 'Koramangala 5th Block'
    },
    {
        'Time': (current_time - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
        'Type': 'Traffic Congestion',
        'Severity': 'Medium',
        'Location': 'Hosur Road Junction'
    },
    {
        'Time': (current_time - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M'),
        'Type': 'Anomaly Detected',
        'Severity': 'Critical',
        'Location': 'Sarjapur Road'
    }
]

alerts_df = pd.DataFrame(alerts_data)
st.dataframe(alerts_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Recommendations
st.header("💡 Health & Policy Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 For Citizens")
    st.info("""
    **Health Advisory (Current AQI: {})**
    
    • Avoid outdoor activities between 8-10 AM and 6-8 PM
    
    • Use N95 masks when going outside
    
    • Keep windows closed during high pollution periods
    
    • Use air purifiers indoors if available
    
    • Stay hydrated and monitor respiratory health
    """.format(current_aqi))

with col2:
    st.subheader("🏛️ For Policymakers")
    st.warning("""
    **Action Items**
    
    • Implement odd-even vehicle scheme during peak hours
    
    • Increase public transport frequency on major routes
    
    • Deploy mobile air purification units in hotspot areas
    
    • Issue public health advisory through official channels
    
    • Monitor construction activities and enforce dust control
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌍 <b>Koramangala Environmental Monitoring System</b></p>
    <p>Made with ❤️ for a cleaner Koramangala | Data updated every 15 minutes</p>
</div>
""", unsafe_allow_html=True)

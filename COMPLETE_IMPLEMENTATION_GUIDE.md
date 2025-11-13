# 🌍 Koramangala Environmental Monitoring System - Complete Implementation Guide

## 📋 Project Overview

This comprehensive guide provides step-by-step instructions for implementing a multi-modal environmental monitoring system for Koramangala, Bengaluru. The system integrates real-time air quality, weather, traffic, satellite imagery, and social media data to provide pollution prediction, anomaly detection, and decision intelligence.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Collection Layer                    │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ IoT      │ Weather  │ Satellite│ Traffic  │ Social Media    │
│ Sensors  │ APIs     │ Imagery  │ APIs     │ Sentiment       │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing & Fusion Layer               │
│  (Cleaning, Normalization, Temporal Alignment, Features)     │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML/DL Models Layer                        │
│  • LSTM/GRU Time Series Forecasting                          │
│  • Transformer-based Sequence Models                         │
│  • CNN + LSTM Multi-modal Fusion                             │
│  • Isolation Forest / LSTM-Autoencoder Anomaly Detection     │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Decision Intelligence Layer                      │
│  • Risk Scoring   • Alerts   • Recommendations               │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           Visualization & Dashboard (Streamlit)              │
│  • Real-time Maps  • Charts  • Alerts  • Predictions         │
└─────────────────────────────────────────────────────────────┘
```

## 📏 Project Structure

```
koramangala-environmental-monitoring/
│
├── data/
│   ├── raw/                      # Raw data from sources
│   ├── processed/                # Cleaned & processed data
│   ├── models/                   # Trained model checkpoints
│   └── cache/                    # Cached API responses
│
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── iot_sensors.py        # IoT sensor data collection
│   │   ├── weather_api.py        # OpenWeather API integration
│   │   ├── air_quality_api.py    # AQICN API integration
│   │   ├── satellite_data.py     # Google Earth Engine
│   │   ├── traffic_api.py        # Google Maps Traffic API
│   │   └── social_media.py       # Twitter/X sentiment analysis
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaning.py      # Missing values, outliers
│   │   ├── normalization.py      # Feature scaling
│   │   ├── temporal_alignment.py # Time synchronization
│   │   └── feature_engineering.py # Feature extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_forecaster.py    # LSTM/GRU models
│   │   ├── transformer_model.py  # Transformer architecture
│   │   ├── multimodal_fusion.py  # CNN+LSTM fusion
│   │   ├── anomaly_detection.py  # Anomaly models
│   │   └── model_utils.py        # Training utilities
│   │
│   ├── decision_intelligence/
│   │   ├── __init__.py
│   │   ├── risk_scoring.py       # Risk calculation
│   │   ├── alert_system.py       # Alert generation
│   │   └── recommendations.py    # Policy recommendations
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── maps.py               # Folium/Leaflet maps
│   │   ├── charts.py             # Plotly charts
│   │   └── dashboard_components.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgres_handler.py   # Time-series data
│   │   ├── mongo_handler.py      # Unstructured data
│   │   └── redis_handler.py      # Caching
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logger.py             # Logging utilities
│       └── constants.py          # Constants
│
├── app/
│   ├── streamlit_app.py          # Main Streamlit dashboard
│   ├── pages/
│   │   ├── 01_🌡️_Air_Quality.py
│   │   ├── 02_📊_Predictions.py
│   │   ├── 03_⚠️_Alerts.py
│   │   └── 04_📝_Reports.py
│   └── config.toml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── test_data_collection.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/
│   ├── API_DOCUMENTATION.md
│   ├── MODEL_ARCHITECTURE.md
│   └── DEPLOYMENT.md
│
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # Project documentation
└── COMPLETE_IMPLEMENTATION_GUIDE.md
```

## 🚀 Phase 1: Problem Understanding & System Design

### Geographic Scope
**Area**: Koramangala, Bengaluru (12.9352° N, 77.6245° E)
- Bounding Box: 
  - North: 12.9485° N
  - South: 12.9219° N
  - East: 77.6394° E
  - West: 77.6096° E

### Environmental Challenges
1. **Air Pollution**: Traffic congestion, industrial emissions
2. **Urban Heat Islands**: Dense construction, limited green space
3. **Noise Pollution**: High traffic volume
4. **Water Quality**: Bellandur Lake proximity

### Key Performance Indicators
- Air Quality Index (AQI) prediction accuracy > 85%
- Pollution hotspot detection within 15 minutes
- False positive alert rate < 5%
- System uptime > 99%

---

## 📊 Phase 2: Multi-Modal Data Collection

### 2.1 Air Quality Data (AQICN API)

**API Endpoint**: `https://api.waqi.info/`

```python
# src/data_collection/air_quality_api.py
import requests
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List

class AirQualityCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.waqi.info"
    
    def get_station_data(self, lat: float, lon: float) -> Dict:
        """Get air quality data for specific coordinates"""
        url = f"{self.base_url}/feed/geo:{lat};{lon}/"
        params = {'token': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                return {
                    'timestamp': datetime.now(),
                    'station_id': data['data'].get('idx'),
                    'aqi': data['data'].get('aqi'),
                    'pm25': data['data']['iaqi'].get('pm25', {}).get('v'),
                    'pm10': data['data']['iaqi'].get('pm10', {}).get('v'),
                    'o3': data['data']['iaqi'].get('o3', {}).get('v'),
                    'no2': data['data']['iaqi'].get('no2', {}).get('v'),
                    'so2': data['data']['iaqi'].get('so2', {}).get('v'),
                    'co': data['data']['iaqi'].get('co', {}).get('v'),
                    'temperature': data['data']['iaqi'].get('t', {}).get('v'),
                    'pressure': data['data']['iaqi'].get('p', {}).get('v'),
                    'humidity': data['data']['iaqi'].get('h', {}).get('v'),
                    'wind_speed': data['data']['iaqi'].get('w', {}).get('v'),
                    'latitude': lat,
                    'longitude': lon
                }
        except Exception as e:
            print(f"Error fetching air quality data: {e}")
            return None
    
    def collect_koramangala_data(self) -> pd.DataFrame:
        """Collect data from multiple points in Koramangala"""
        # Grid of monitoring points
        points = [
            (12.9352, 77.6245),  # Central Koramangala
            (12.9412, 77.6281),  # North
            (12.9292, 77.6209),  # South
            (12.9365, 77.6350),  # East
            (12.9339, 77.6140),  # West
        ]
        
        all_data = []
        for lat, lon in points:
            data = self.get_station_data(lat, lon)
            if data:
                all_data.append(data)
        
        return pd.DataFrame(all_data)

# Usage
if __name__ == "__main__":
    collector = AirQualityCollector(api_key=os.getenv('AQICN_API_KEY'))
    df = collector.collect_koramangala_data()
    df.to_csv('data/raw/air_quality_koramangala.csv', index=False)
```

### 2.2 Weather Data (OpenWeather API)

```python
# src/data_collection/weather_api.py
import requests
import pandas as pd
from datetime import datetime
import os

class WeatherCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Get current weather data"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return {
                'timestamp': datetime.fromtimestamp(data['dt']),
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'temp_min': data['main']['temp_min'],
                'temp_max': data['main']['temp_max'],
                'pressure': data['main']['pressure'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg'),
                'wind_gust': data['wind'].get('gust'),
                'clouds': data['clouds']['all'],
                'visibility': data.get('visibility'),
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'rain_1h': data.get('rain', {}).get('1h', 0),
                'latitude': lat,
                'longitude': lon
            }
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_forecast(self, lat: float, lon: float) -> pd.DataFrame:
        """Get 5-day weather forecast"""
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            forecast_data = []
            for item in data['list']:
                forecast_data.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'clouds': item['clouds']['all'],
                    'weather_main': item['weather'][0]['main'],
                    'rain_3h': item.get('rain', {}).get('3h', 0)
                })
            
            return pd.DataFrame(forecast_data)
        except Exception as e:
            print(f"Error fetching forecast data: {e}")
            return pd.DataFrame()
```

### 2.3 Satellite Imagery (Google Earth Engine)

```python
# src/data_collection/satellite_data.py
import ee
import geemap
import pandas as pd
from datetime import datetime, timedelta

# Initialize Earth Engine
ee.Initialize()

class SatelliteDataCollector:
    def __init__(self):
        self.koramangala_bounds = ee.Geometry.Rectangle([77.6096, 12.9219, 77.6394, 12.9485])
    
    def get_sentinel2_data(self, start_date: str, end_date: str):
        """Get Sentinel-2 satellite imagery"""
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(self.koramangala_bounds)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        return collection
    
    def calculate_ndvi(self, image):
        """Calculate Normalized Difference Vegetation Index"""
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    def get_air_quality_indicators(self, start_date: str, end_date: str):
        """Extract air quality indicators from satellite data"""
        # Sentinel-5P NO2 data
        no2_collection = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
                          .select('NO2_column_number_density')
                          .filterBounds(self.koramangala_bounds)
                          .filterDate(start_date, end_date))
        
        # Sentinel-5P CO data
        co_collection = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO')
                         .select('CO_column_number_density')
                         .filterBounds(self.koramangala_bounds)
                         .filterDate(start_date, end_date))
        
        # Aerosol data
        aerosol_collection = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI')
                              .select('absorbing_aerosol_index')
                              .filterBounds(self.koramangala_bounds)
                              .filterDate(start_date, end_date))
        
        return {
            'no2': no2_collection,
            'co': co_collection,
            'aerosol': aerosol_collection
        }
```

### 2.4 Traffic Data (Google Maps API)

```python
# src/data_collection/traffic_api.py
import googlemaps
import pandas as pd
from datetime import datetime

class TrafficDataCollector:
    def __init__(self, api_key: str):
        self.gmaps = googlemaps.Client(key=api_key)
        self.major_roads = [
            {"name": "Hosur Road", "coords": [(12.9352, 77.6245), (12.9412, 77.6281)]},
            {"name": "Sarjapur Road", "coords": [(12.9292, 77.6209), (12.9365, 77.6350)]},
            {"name": "Koramangala 80 Feet Road", "coords": [(12.9339, 77.6140), (12.9352, 77.6245)]}
        ]
    
    def get_traffic_conditions(self):
        """Get current traffic conditions"""
        traffic_data = []
        
        for road in self.major_roads:
            try:
                origin = road['coords'][0]
                destination = road['coords'][1]
                
                # Get directions with traffic model
                directions = self.gmaps.directions(
                    origin=origin,
                    destination=destination,
                    mode="driving",
                    departure_time=datetime.now(),
                    traffic_model="best_guess"
                )
                
                if directions:
                    leg = directions[0]['legs'][0]
                    traffic_data.append({
                        'timestamp': datetime.now(),
                        'road_name': road['name'],
                        'distance_meters': leg['distance']['value'],
                        'duration_seconds': leg['duration']['value'],
                        'duration_in_traffic_seconds': leg.get('duration_in_traffic', {}).get('value'),
                        'traffic_delay': leg.get('duration_in_traffic', {}).get('value', 0) - leg['duration']['value']
                    })
            except Exception as e:
                print(f"Error getting traffic data for {road['name']}: {e}")
        
        return pd.DataFrame(traffic_data)
```

---

## 🧹 Phase 3: Data Preprocessing & Fusion

```python
# src/preprocessing/data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

class DataCleaner:
    def __init__(self):
        self.imputer = KNNImputer(n_neighbors=5)
    
    def remove_outliers(self, df: pd.DataFrame, columns: list, method='iqr', threshold=3):
        """Remove outliers using IQR or Z-score method"""
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[z_scores < threshold]
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, strategy='knn'):
        """Handle missing values"""
        if strategy == 'knn':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        elif strategy == 'interpolate':
            df = df.interpolate(method='time', limit_direction='both')
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame):
        """Validate data quality and generate report"""
        report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict()
        }
        return report

# src/preprocessing/temporal_alignment.py
class TemporalAligner:
    def __init__(self, target_frequency='15min'):
        self.target_frequency = target_frequency
    
    def align_timestamps(self, datasets: dict):
        """Align multiple datasets to common timestamp"""
        aligned_data = {}
        
        # Find common time range
        min_time = max([df['timestamp'].min() for df in datasets.values()])
        max_time = min([df['timestamp'].max() for df in datasets.values()])
        
        # Create common time index
        common_index = pd.date_range(start=min_time, end=max_time, freq=self.target_frequency)
        
        # Resample each dataset
        for name, df in datasets.items():
            df = df.set_index('timestamp')
            df_resampled = df.resample(self.target_frequency).mean()
            df_resampled = df_resampled.reindex(common_index)
            aligned_data[name] = df_resampled
        
        return aligned_data

# src/preprocessing/feature_engineering.py
class FeatureEngineer:
    def create_temporal_features(self, df: pd.DataFrame):
        """Create temporal features"""
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: list, lags=[1, 2, 3, 6, 12, 24]):
        """Create lagged features"""
        df = df.copy()
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: list, windows=[3, 6, 12, 24]):
        """Create rolling statistics"""
        df = df.copy()
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
        return df
```

---

## 🧠 Phase 5: Machine Learning & Deep Learning Models

### 5.1 LSTM Time Series Forecasting

```python
# src/models/lstm_forecaster.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=24, forecast_horizon=1):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred = self.model(batch_x)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)
```

### 5.2 Transformer Model

```python
# src/models/transformer_model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerForecaster(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, output_size=1):
        super(TransformerForecaster, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # Use last timestep for prediction
        x = x[:, -1, :]
        output = self.fc(x)
        return output
```

### 5.3 Multi-Modal Fusion Model

```python
# src/models/multimodal_fusion.py
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting features from satellite imagery"""
    def __init__(self, in_channels=3, feature_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(128, feature_dim)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for multi-modal fusion"""
    def __init__(self, feature_dim=128, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
    
    def forward(self, query, key, value):
        # Cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        x = self.norm(query + attn_output)
        # Feed-forward
        ffn_output = self.ffn(x)
        output = self.norm(x + ffn_output)
        return output

class MultiModalFusionModel(nn.Module):
    def __init__(self, timeseries_input_size, image_channels=3, 
                 hidden_size=128, num_lstm_layers=2, output_size=1):
        super(MultiModalFusionModel, self).__init__()
        
        # Time series branch (LSTM)
        self.lstm = nn.LSTM(
            input_size=timeseries_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # Image branch (CNN)
        self.cnn = CNNFeatureExtractor(in_channels=image_channels, feature_dim=hidden_size)
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(feature_dim=hidden_size)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
    
    def forward(self, timeseries_data, image_data):
        # Extract time series features
        lstm_out, _ = self.lstm(timeseries_data)
        timeseries_features = lstm_out[:, -1, :].unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Extract image features
        image_features = self.cnn(image_data).unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Cross-modal fusion
        fused_features = self.fusion(
            query=timeseries_features,
            key=image_features,
            value=image_features
        )
        
        # Prediction
        output = self.fc(fused_features.squeeze(1))
        return output
```

### 5.4 Anomaly Detection

```python
# src/models/anomaly_detection.py
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import numpy as np

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for anomaly detection"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
    
    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Prepare decoder input
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decode
        reconstructed, _ = self.decoder(decoder_input)
        
        return reconstructed
    
    def detect_anomalies(self, x, threshold_percentile=95):
        """Detect anomalies based on reconstruction error"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            reconstruction_error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        
        threshold = np.percentile(reconstruction_error.cpu().numpy(), threshold_percentile)
        anomalies = reconstruction_error > threshold
        
        return anomalies, reconstruction_error

class AnomalyDetector:
    def __init__(self, method='isolation_forest'):
        self.method = method
        if method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif method == 'lstm_autoencoder':
            self.model = None  # Will be initialized with data
    
    def fit(self, X):
        """Fit the anomaly detection model"""
        if self.method == 'isolation_forest':
            self.model.fit(X)
        elif self.method == 'lstm_autoencoder':
            # Initialize and train LSTM autoencoder
            input_size = X.shape[2] if len(X.shape) == 3 else X.shape[1]
            self.model = LSTMAutoencoder(input_size=input_size)
            # Training code here
    
    def predict(self, X):
        """Predict anomalies"""
        if self.method == 'isolation_forest':
            predictions = self.model.predict(X)
            return predictions == -1  # -1 indicates anomaly
        elif self.method == 'lstm_autoencoder':
            anomalies, scores = self.model.detect_anomalies(torch.FloatTensor(X))
            return anomalies.numpy(), scores.numpy()
```

---

## 🎯 Phase 6-7: Decision Intelligence & Visualization

### Streamlit Dashboard

```python
# app/streamlit_app.py
import streamlit as st
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Koramangala Environmental Monitor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Koramangala Environmental Monitoring System")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    view_mode = st.selectbox("View Mode", ["Real-time", "Historical", "Predictions"])
    time_range = st.slider("Time Range (hours)", 1, 168, 24)
    refresh_rate = st.number_input("Refresh Rate (seconds)", 5, 300, 60)

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current AQI", "156", "+12", delta_color="inverse")
with col2:
    st.metric("Temperature", "28°C", "+2°C")
with col3:
    st.metric("Active Alerts", "3", "+1", delta_color="inverse")
with col4:
    st.metric("PM2.5", "78 µg/m³", "+8", delta_color="inverse")

# Map
st.header("🗺️ Real-time Air Quality Map")

# Create Folium map centered on Koramangala
m = folium.Map(
    location=[12.9352, 77.6245],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Add heatmap layer
from folium.plugins import HeatMap

heat_data = [
    [12.9352, 77.6245, 0.8],
    [12.9412, 77.6281, 0.6],
    [12.9292, 77.6209, 0.9],
    [12.9365, 77.6350, 0.7],
    [12.9339, 77.6140, 0.5],
]

HeatMap(heat_data).add_to(m)

folium_static(m)

# Time series charts
st.header("📊 Temporal Analysis")

col1, col2 = st.columns(2)

with col1:
    # AQI time series
    dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
    aqi_data = np.random.randint(50, 200, 168)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=aqi_data,
        mode='lines',
        name='AQI',
        line=dict(color='rgb(255, 127, 14)', width=2)
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy")
    
    fig.update_layout(title="Air Quality Index Trend", xaxis_title="Time", yaxis_title="AQI")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # PM2.5 and PM10
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=np.random.randint(30, 100, 168), name='PM2.5'))
    fig.add_trace(go.Scatter(x=dates, y=np.random.randint(50, 150, 168), name='PM10'))
    fig.update_layout(title="Particulate Matter Levels", xaxis_title="Time", yaxis_title="µg/m³")
    st.plotly_chart(fig, use_container_width=True)

# Predictions
st.header("🔮 Future Predictions")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Next 6 Hours")
    predictions_6h = pd.DataFrame({
        'Hour': [f'+{i}h' for i in range(1, 7)],
        'AQI': np.random.randint(140, 180, 6),
        'Confidence': np.random.uniform(0.85, 0.95, 6)
    })
    st.dataframe(predictions_6h)

with col2:
    st.subheader("Next 24 Hours")
    fig = px.bar(x=[f'+{i}h' for i in range(24)], y=np.random.randint(100, 200, 24))
    fig.update_layout(xaxis_title="Time", yaxis_title="Predicted AQI")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Weekly Forecast")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=days, y=np.random.randint(120, 180, 7), name='AQI Range'))
    fig.update_layout(xaxis_title="Day", yaxis_title="Predicted AQI")
    st.plotly_chart(fig, use_container_width=True)

# Alerts
st.header("⚠️ Active Alerts")

alerts_df = pd.DataFrame({
    'Time': [datetime.now() - timedelta(hours=i) for i in range(3)],
    'Type': ['High PM2.5', 'Traffic Congestion', 'Anomaly Detected'],
    'Severity': ['High', 'Medium', 'Critical'],
    'Location': ['Koramangala 5th Block', 'Hosur Road', 'Sarjapur Road']
})

st.dataframe(alerts_df, use_container_width=True)

# Recommendations
st.header("💡 Recommendations")

st.info("""
**For Citizens:**
- Avoid outdoor activities during peak hours (8-10 AM, 6-8 PM)
- Use N95 masks if AQI > 150
- Keep windows closed during high pollution periods

**For Policymakers:**
- Implement odd-even vehicle scheme during peak pollution days
- Increase public transport frequency on major routes
- Deploy mobile air purification units in hotspot areas
""")
```

---

## 📦 Requirements & Setup

### requirements.txt

```txt
# Core
python>=3.9
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Data Collection
requests>=2.31.0
googlemaps>=4.10.0
earth engineapi>=0.1.360
geemap>=0.28.0
tweepy>=4.14.0

# Machine Learning
scikit-learn>=1.3.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# Deep Learning Utilities
tensorboard>=2.13.0
wandb>=0.15.0

# Data Processing
opencv-python>=4.8.0
Pillow>=10.0.0
pyproj>=3.6.0

# Visualization
streamlit>=1.25.0
plotly>=5.15.0
folium>=0.14.0
streamlit-folium>=0.13.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Database
psycopg2-binary>=2.9.6
pymongo>=4.4.0
redis>=4.6.0
sqlalchemy>=2.0.0

# API & Web
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
joblib>=1.3.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Deployment
docker>=6.1.0
gunicorn>=21.2.0
```

### .env.example

```bash
# API Keys
AQICN_API_KEY=your_aqicn_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here

# Google Earth Engine
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=koramangala_env
POSTGRES_USER=admin
POSTGRES_PASSWORD=your_password

MONGO_URI=mongodb://localhost:27017/
REDIS_HOST=localhost
REDIS_PORT=6379

# ML Model Paths
MODEL_CHECKPOINT_DIR=./data/models/
CACHE_DIR=./data/cache/

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/SmrithiWarrier/koramangala-environmental-monitoring.git
cd koramangala-environmental-monitoring

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Initialize databases
python scripts/init_databases.py

# 6. Collect initial data
python scripts/collect_data.py --days 30

# 7. Train models
python scripts/train_models.py

# 8. Run Streamlit dashboard
streamlit run app/streamlit_app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access dashboard at http://localhost:8501
```

---

## ✅ Next Steps

1. **Create directory structure**: Run `python scripts/create_project_structure.py`
2. **Obtain API keys**: Register for all required API services
3. **Set up databases**: Install PostgreSQL, MongoDB, Redis
4. **Collect baseline data**: Run data collection for 30 days minimum
5. **Train initial models**: Use collected data to train forecasting models
6. **Deploy dashboard**: Launch Streamlit app for visualization
7. **Set up automated collection**: Configure cron jobs for continuous data ingestion
8. **Monitor and iterate**: Track model performance and refine

---

## 📚 Additional Resources

- **API Documentation**: See `docs/API_DOCUMENTATION.md`
- **Model Architecture**: See `docs/MODEL_ARCHITECTURE.md`
- **Deployment Guide**: See `docs/DEPLOYMENT.md`
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`

---

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributors

Smrithi Warrier (@SmrithiWarrier)

---

**Note**: This is a comprehensive guide. Implement phase by phase, testing each component before moving to the next. Prioritize data collection and preprocessing before diving into complex ML models.

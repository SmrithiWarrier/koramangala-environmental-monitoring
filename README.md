# 🌍 Koramangala Environmental Monitoring System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

> 🏆 **Comprehensive Multi-Modal Environmental Monitoring System** for Koramangala, Bengaluru - Real-time pollution prediction, traffic analysis, and urban environmental intelligence platform with ML/DL models, interactive dashboards, and decision intelligence.

##  📊 Overview

This project implements an advanced environmental monitoring and prediction system for the Koramangala region of Bengaluru. It integrates multiple data sources including air quality sensors, weather data, satellite imagery, traffic information, and social media sentiment to provide:

- **Real-time Air Quality Monitoring** - Track PM2.5, PM10, NO2, SO2, CO, and O3 levels
- **Predictive Analytics** - LSTM/Transformer models for pollution forecasting (6h, 24h, 7-day)
- **Anomaly Detection** - Identify unusual pollution events using LSTM Autoencoders
- **Multi-Modal Data Fusion** - Combine satellite imagery with time-series data using CNN+LSTM+Cross-Attention
- **Traffic Correlation Analysis** - Link congestion patterns with air quality
- **Interactive Dashboards** - Streamlit-based visualization with maps, charts, and alerts
- **Decision Intelligence** - Automated alerts and policy recommendations

## 🎯 Key Features

### Data Collection
- 🌡️ **Air Quality**: AQICN API integration
- ☁️ **Weather Data**: OpenWeather API (temperature, humidity, wind, rain)
- 🛰️ **Satellite Imagery**: Google Earth Engine (Sentinel-2, Sentinel-5P)
- 🚗 **Traffic Data**: Google Maps Traffic API
- 📱 **Social Media**: Twitter/X sentiment analysis

### Machine Learning Models
- **LSTM Forecaster**: Time-series prediction for pollution levels
- **Transformer Model**: Advanced sequence-to-sequence forecasting
- **Multi-Modal Fusion**: CNN+LSTM with cross-attention mechanism
- **Anomaly Detection**: Isolation Forest and LSTM Autoencoder
- **GNN+LSTM**: Graph Neural Networks for spatial-temporal modeling

### Visualization & Dashboard
- Interactive maps with heatmaps (Folium)
- Time-series charts (Plotly)
- Real-time metrics and KPIs
- Alert system with severity levels
- Policy recommendations

## 🛠️ Technology Stack

- **Language**: Python 3.9+
- **Deep Learning**: PyTorch 2.0+, Transformers
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Streamlit, Plotly, Folium
- **Databases**: PostgreSQL (time-series), MongoDB (metadata), Redis (caching)
- **APIs**: OpenWeather, AQICN, Google Maps, Google Earth Engine
- **Deployment**: Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.9 or higher
- PostgreSQL 13+
- MongoDB 5.0+
- Redis 6.0+
- 8GB RAM minimum
- GPU (optional, for faster training)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SmrithiWarrier/koramangala-environmental-monitoring.git
cd koramangala-environmental-monitoring

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# 5. Initialize databases (ensure PostgreSQL, MongoDB, Redis are running)
python scripts/init_databases.py

# 6. Collect initial data (30 days recommended)
python scripts/collect_data.py --days 30

# 7. Train models
python scripts/train_models.py

# 8. Launch dashboard
streamlit run app/streamlit_app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access dashboard at http://localhost:8501
```

## 📁 Project Structure

```
koramangala-environmental-monitoring/
├── data/                      # Data storage
│   ├── raw/                   # Raw collected data
│   ├── processed/             # Cleaned & processed data
│   ├── models/                # Trained model checkpoints
│   └── cache/                 # Cached API responses
├── src/                       # Source code
│   ├── data_collection/      # API integrations
│   ├── preprocessing/        # Data cleaning & feature engineering
│   ├── models/                # ML/DL model implementations
│   ├── decision_intelligence/ # Alerts & recommendations
│   ├── visualization/        # Charts & maps
│   └── database/              # Database handlers
├── app/                       # Streamlit dashboard
├── notebooks/                 # Jupyter notebooks for EDA
├── tests/                     # Unit tests
├── docker/                    # Docker configurations
└── docs/                      # Documentation
```

## 📊 Data Sources

| Source | API | Update Frequency | Key Metrics |
|--------|-----|------------------|-------------|
| AQICN | https://aqicn.org | 15 min | PM2.5, PM10, NO2, SO2, CO, O3, AQI |
| OpenWeather | https://openweathermap.org | 30 min | Temperature, Humidity, Wind, Pressure |
| Google Earth Engine | https://earthengine.google.com | Daily | NDVI, NO2, CO, Aerosol Index |
| Google Maps | https://developers.google.com/maps | 10 min | Traffic delays, congestion levels |
| Twitter | https://developer.twitter.com | 1 hour | Environmental sentiment |

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- Missing value imputation (KNN)
- Outlier removal (IQR/Z-score)
- Temporal alignment (15-min intervals)
- Feature engineering (lag, rolling, cyclical)

### 2. Model Training
```python
# LSTM Forecaster
- Sequence length: 24 hours
- Hidden size: 128
- Num layers: 2
- Dropout: 0.2

# Transformer
- d_model: 128
- Num heads: 8
- Num encoder layers: 3

# Multi-Modal Fusion
- CNN (satellite) + LSTM (time-series)
- Cross-attention mechanism
```

### 3. Evaluation Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- Direction Accuracy

## 📝 API Keys Required

To run this project, you'll need to obtain free API keys from:

1. **AQICN**: https://aqicn.org/data-platform/token/
2. **OpenWeather**: https://openweathermap.org/api
3. **Google Cloud** (for Maps & Earth Engine): https://console.cloud.google.com/
4. **Twitter Developer**: https://developer.twitter.com/ (optional)

Add these keys to your `.env` file (see `.env.example` for template).

## 📚 Documentation

- **Complete Implementation Guide**: See [COMPLETE_IMPLEMENTATION_GUIDE.md](COMPLETE_IMPLEMENTATION_GUIDE.md)
- **Model Architecture**: `docs/MODEL_ARCHITECTURE.md` (coming soon)
- **API Documentation**: `docs/API_DOCUMENTATION.md` (coming soon)
- **Deployment Guide**: `docs/DEPLOYMENT.md` (coming soon)

## 🔬 Roadmap

- [ ] Real-time data pipeline with Apache Kafka
- [ ] Mobile app (React Native)
- [ ] Advanced GNN models for spatial correlation
- [ ] Integration with CPCB (Central Pollution Control Board) data
- [ ] Citizen science IoT sensor network
- [ ] API for third-party integrations

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AQICN for air quality data
- OpenWeather for weather data
- Google Earth Engine for satellite imagery
- Streamlit for the amazing dashboard framework

## 📧 Contact

**Smrithi Warrier** - [@SmrithiWarrier](https://github.com/SmrithiWarrier)

Project Link: [https://github.com/SmrithiWarrier/koramangala-environmental-monitoring](https://github.com/SmrithiWarrier/koramangala-environmental-monitoring)

---

**Made with ❤️ for a cleaner Koramangala**

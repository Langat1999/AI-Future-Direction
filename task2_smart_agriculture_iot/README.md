# Smart Agriculture IoT Simulation with AI

An AI-driven IoT simulation system for smart farming featuring multi-sensor data generation, crop health prediction, and automated decision-making.

## Overview

This project simulates a comprehensive smart agriculture system that combines:
- **IoT Sensor Simulation**: Temperature, humidity, soil moisture, light, electrical conductivity (EC), rainfall
- **AI Prediction Models**: Crop health classification, yield prediction, irrigation recommendations
- **Automated Alerts**: Real-time notifications for stress conditions and optimal interventions
- **Visualization Dashboard**: Interactive plots and metrics

## Features

- **Realistic Sensor Simulation**: Physics-based models with daily/seasonal patterns and noise
- **Time-Series Data Generation**: Synthetic multi-year agricultural data
- **Machine Learning Models**:
  - Random Forest for crop health classification
  - Gradient Boosting for yield prediction
  - Rule-based decision system for irrigation/fertilization
- **Interactive Visualization**: Real-time sensor dashboards and historical trends
- **Alert System**: Automated notifications for drought stress, nutrient deficiency, pest risk
- **Scenario Testing**: Pre-defined scenarios (optimal, drought, nutrient deficiency)

## System Architecture

```
┌─────────────────┐
│  IoT Sensors    │ → Soil Moisture, Temp, Humidity, Light, EC, Rainfall
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Data Generator  │ → Synthetic time-series with realistic patterns
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  AI Predictor   │ → Crop Health, Yield Prediction
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Alert System   │ → Irrigation, Fertilizer, Pest Alerts
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Visualization  │ → Dashboard, Historical Plots
└─────────────────┘
```

## Project Structure

```
task2_smart_agriculture_iot/
├── README.md
├── requirements.txt
├── agriculture_simulation.py    # Main simulation script
├── agriculture_demo.ipynb       # Interactive demo notebook
├── REPORT.md                    # Technical report
├── src/
│   ├── config.py               # Configuration
│   ├── iot_sensors.py          # Sensor simulation classes
│   ├── ai_predictor.py         # ML prediction models
│   ├── alert_system.py         # Alert logic
│   ├── data_generator.py       # Synthetic data generation
│   └── visualization.py        # Dashboard and plotting
├── models/
│   ├── crop_health_model.pkl   # Trained crop health classifier
│   ├── yield_model.pkl         # Trained yield predictor
│   └── scaler.pkl              # Feature scaler
├── data/
│   ├── sensor_data.csv         # Generated sensor readings
│   ├── predictions.csv         # Model predictions
│   └── alerts.csv              # Alert history
└── results/
    ├── visualizations/         # Generated plots
    └── simulation_logs/        # Simulation output logs
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
cd task2_smart_agriculture_iot
pip install -r requirements.txt
```

## Usage

### Option 1: Run Main Simulation

```bash
python agriculture_simulation.py --days 365 --scenario optimal
```

**Parameters:**
- `--days`: Number of days to simulate (default: 365)
- `--scenario`: Pre-defined scenario (optimal, drought, nutrient_deficiency, random)
- `--seed`: Random seed for reproducibility
- `--output`: Output directory for results

**Example:**
```bash
# Simulate 1 year of optimal conditions
python agriculture_simulation.py --days 365 --scenario optimal

# Simulate drought stress
python agriculture_simulation.py --days 180 --scenario drought

# Random conditions with seed
python agriculture_simulation.py --days 365 --scenario random --seed 42
```

### Option 2: Interactive Jupyter Notebook

```bash
jupyter notebook agriculture_demo.ipynb
```

The notebook includes:
- Sensor data visualization
- AI model training and testing
- Alert system demonstration
- Interactive dashboard
- Scenario comparison

### Option 3: Use as Python Module

```python
from src.iot_sensors import SoilMoistureSensor, TemperatureSensor
from src.ai_predictor import CropHealthPredictor
from src.alert_system import AlertSystem

# Create sensors
soil_sensor = SoilMoistureSensor()
temp_sensor = TemperatureSensor()

# Generate readings
moisture = soil_sensor.read()
temperature = temp_sensor.read()

# Predict crop health
predictor = CropHealthPredictor()
health_status = predictor.predict(features)

# Check for alerts
alert_system = AlertSystem()
alerts = alert_system.check_conditions(sensor_data)
```

## Sensor Specifications

### 1. Soil Moisture Sensor
- **Range**: 0-100% volumetric water content
- **Optimal**: 40-60% for most crops
- **Update Frequency**: Every hour
- **Model**: Simple water balance (rainfall + irrigation - evaporation - drainage)

### 2. Temperature Sensor (Air)
- **Range**: -10°C to 50°C
- **Optimal**: 20-30°C for most crops
- **Update Frequency**: Every 10 minutes
- **Model**: Sinusoidal daily pattern + seasonal variation + noise

### 3. Humidity Sensor
- **Range**: 30-100% relative humidity
- **Optimal**: 50-70%
- **Update Frequency**: Every 10 minutes
- **Model**: Correlated with temperature (inverse relationship)

### 4. Light Sensor (PAR - Photosynthetically Active Radiation)
- **Range**: 0-2000 µmol/m²/s
- **Optimal**: >400 µmol/m²/s for most crops
- **Update Frequency**: Every 10 minutes
- **Model**: Day/night cycle with cloud cover simulation

### 5. Electrical Conductivity (EC) Sensor
- **Range**: 0-5 dS/m
- **Optimal**: 1-3 dS/m (crop-dependent)
- **Update Frequency**: Daily
- **Model**: Depletes over time (nutrient uptake), replenished with fertilization

### 6. Rain Gauge
- **Range**: 0-100 mm/day
- **Update Frequency**: Daily
- **Model**: Stochastic events with seasonal patterns

## AI Models

### 1. Crop Health Classifier

**Type**: Random Forest Classifier

**Classes**:
- Excellent (4)
- Good (3)
- Fair (2)
- Poor (1)
- Critical (0)

**Features**:
- Soil moisture
- Temperature
- Humidity
- Light (PAR)
- EC (nutrient proxy)
- Days since planting
- Recent rainfall

**Performance**:
- Accuracy: ~88%
- Balanced across classes

### 2. Yield Predictor

**Type**: Gradient Boosting Regressor

**Output**: Estimated yield (kg/ha)

**Features**:
- Aggregate sensor statistics (mean, std, min, max over season)
- Growth stage indicators
- Stress event counts
- Cumulative rainfall

**Performance**:
- R² Score: ~0.82
- RMSE: ~450 kg/ha

### 3. Irrigation Recommender

**Type**: Rule-based decision system

**Logic**:
```python
if soil_moisture < 30%:
    recommendation = "Irrigate URGENTLY (drought stress)"
elif soil_moisture < 40%:
    recommendation = "Irrigate soon"
elif soil_moisture > 70%:
    recommendation = "Reduce irrigation (overwatering risk)"
else:
    recommendation = "No action needed"
```

## Alert System

### Alert Types

1. **Drought Stress**
   - Trigger: Soil moisture < 30% for 2+ days
   - Priority: High
   - Action: Immediate irrigation

2. **Nutrient Deficiency**
   - Trigger: EC < 1.0 dS/m
   - Priority: Medium
   - Action: Fertilizer application

3. **Heat Stress**
   - Trigger: Temperature > 35°C for 3+ hours
   - Priority: High
   - Action: Increase irrigation, shade if possible

4. **Disease Risk**
   - Trigger: High humidity (>80%) + moderate temp (20-25°C)
   - Priority: Medium
   - Action: Monitor for fungal diseases

5. **Frost Warning**
   - Trigger: Temperature forecast < 5°C
   - Priority: High
   - Action: Protective measures

### Alert Logging

All alerts are logged to `data/alerts.csv`:
```csv
timestamp,alert_type,severity,message,sensor_values,recommended_action
2025-06-15 08:00:00,drought_stress,high,Soil moisture at 28%,"{...}",Irrigate 20mm
```

## Visualization

### Dashboard Components

1. **Real-Time Sensor Panel**
   - Current readings for all sensors
   - Color-coded status (green/yellow/red)

2. **Historical Trends**
   - 7-day, 30-day, season-long trends
   - Overlay optimal ranges

3. **Crop Health Timeline**
   - Daily health status
   - Correlation with environmental factors

4. **Alert History**
   - Chronological alert list
   - Alert frequency analysis

5. **Yield Prediction**
   - Current season estimate
   - Confidence interval
   - Comparison to historical averages

### Example Plots

```python
from src.visualization import AgricultureDashboard

dashboard = AgricultureDashboard(sensor_data, predictions, alerts)
dashboard.plot_sensor_timeline()
dashboard.plot_crop_health_heatmap()
dashboard.plot_alert_summary()
dashboard.show()
```

## Scenarios

### 1. Optimal Conditions
- Consistent rainfall (40-60mm/week)
- Moderate temperatures (20-28°C)
- Good soil moisture (45-55%)
- Adequate nutrients (EC 2-3 dS/m)
- **Expected**: Excellent crop health, high yield

### 2. Drought Stress
- Low rainfall (<20mm/month)
- High temperatures (30-38°C)
- Declining soil moisture (20-35%)
- **Expected**: Poor crop health, low yield, many irrigation alerts

### 3. Nutrient Deficiency
- Optimal water/temp
- Low EC (<1.5 dS/m) declining over time
- **Expected**: Fair crop health, moderate yield, fertilizer alerts

### 4. Random (Realistic)
- Variable weather patterns
- Mixed stress events
- **Expected**: Good to fair health, realistic yield

## Configuration

Edit `src/config.py` to customize:

```python
# Simulation parameters
SIMULATION_START_DATE = "2025-01-01"
CROP_TYPE = "corn"  # corn, wheat, soybean
FIELD_AREA_HA = 10

# Sensor thresholds
SOIL_MOISTURE_OPTIMAL = (40, 60)  # %
TEMPERATURE_OPTIMAL = (20, 30)    # °C
EC_OPTIMAL = (1.5, 3.0)           # dS/m

# Alert settings
ALERT_CHECK_FREQUENCY = "hourly"
ALERT_NOTIFICATION_METHOD = "log"  # log, email, sms
```

## Performance

**Data Generation Speed**:
- 1 year (365 days): ~2 seconds
- 10 years: ~20 seconds

**Model Inference**:
- Crop health prediction: <10ms per sample
- Yield prediction: <5ms per sample

**Dashboard Rendering**:
- Interactive plots: ~1 second
- Historical analysis: ~3 seconds

## Example Output

```
==========================================================
Smart Agriculture IoT Simulation
==========================================================
Configuration:
  Duration: 365 days
  Scenario: optimal
  Crop: corn
  Field area: 10 ha

Generating sensor data...
✓ Generated 8,760 hourly readings

Training AI models...
✓ Crop health classifier trained (accuracy: 88.5%)
✓ Yield predictor trained (R²: 0.82)

Running simulation...
Day 1: Health=Excellent, Alerts=0
Day 30: Health=Excellent, Alerts=0
Day 60: Health=Good, Alerts=1 (Fertilizer recommended)
...
Day 365: Health=Good, Alerts=0

==========================================================
Simulation Summary
==========================================================
Total Alerts: 12
  - Drought stress: 2
  - Nutrient deficiency: 8
  - Heat stress: 1
  - Disease risk: 1

Average Crop Health: Good (3.2/4.0)
Predicted Yield: 8,450 kg/ha
Irrigation Events: 15 (total: 300mm)
Fertilizer Applications: 4

Recommendations:
  ✓ Increase fertilizer frequency (current: every 90 days)
  ✓ Monitor for nutrient leaching after heavy rain
  ✓ Overall good management, maintain current practices

Results saved to: results/simulation_2025_11_17/
==========================================================
```

## Troubleshooting

**Issue**: "No module named 'src'"
- **Solution**: Run from project root directory

**Issue**: Low yield predictions
- **Solution**: Check sensor data for stress conditions, adjust scenario

**Issue**: Too many alerts
- **Solution**: Adjust thresholds in `config.py`

## Future Enhancements

- [ ] Integration with real weather API for realistic forecasts
- [ ] LSTM/GRU time-series models for yield prediction
- [ ] Multi-field management
- [ ] Economic optimization (water/fertilizer cost vs. yield)
- [ ] Real hardware sensor integration (MQTT protocol)
- [ ] Web-based dashboard (Dash/Streamlit)

## License

Educational use - MIT License

## Acknowledgments

- Agricultural best practices from FAO guidelines
- Sensor specifications from industry datasheets
- ML models inspired by precision agriculture research

---

For detailed methodology and results, see [REPORT.md](REPORT.md)

For interactive exploration, open [agriculture_demo.ipynb](agriculture_demo.ipynb)

"""
Configuration for Smart Agriculture IoT Simulation
Centralized settings for sensors, models, and simulation parameters
"""

import os
from datetime import datetime, timedelta

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')
LOGS_DIR = os.path.join(RESULTS_DIR, 'simulation_logs')

# ============================================================================
# Simulation Parameters
# ============================================================================
# Time settings
SIMULATION_START_DATE = datetime(2025, 1, 1)
SIMULATION_DAYS = 365
HOURLY_READINGS = True  # True for hourly, False for daily only

# Crop settings
CROP_TYPE = 'corn'  # Options: 'corn', 'wheat', 'soybean', 'rice'
CROP_VARIETY = 'generic'
PLANTING_DATE = SIMULATION_START_DATE + timedelta(days=15)  # Jan 15
HARVEST_DATE = SIMULATION_START_DATE + timedelta(days=180)  # Jul 1
GROWTH_PERIOD_DAYS = 165

# Field settings
FIELD_AREA_HA = 10.0  # hectares
LATITUDE = 40.0  # degrees (affects day length, seasonality)
LONGITUDE = -95.0  # degrees

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Sensor Configuration
# ============================================================================

# Soil Moisture Sensor (Volumetric Water Content)
SOIL_MOISTURE_CONFIG = {
    'min_value': 0.0,      # 0% (completely dry)
    'max_value': 100.0,    # 100% (saturated)
    'optimal_min': 40.0,   # Optimal range lower bound
    'optimal_max': 60.0,   # Optimal range upper bound
    'critical_low': 30.0,  # Below this triggers drought alert
    'critical_high': 75.0, # Above this triggers overwatering alert
    'initial_value': 50.0,
    'noise_std': 2.0,      # Standard deviation of measurement noise
    'update_frequency': '1H'  # Pandas frequency string
}

# Temperature Sensor (Air Temperature in Celsius)
TEMPERATURE_CONFIG = {
    'min_value': -10.0,
    'max_value': 50.0,
    'optimal_min': 20.0,
    'optimal_max': 30.0,
    'critical_low': 5.0,   # Frost risk
    'critical_high': 35.0, # Heat stress
    'daily_mean': 22.0,    # Average daily temperature
    'daily_amplitude': 8.0, # Day-night variation
    'seasonal_amplitude': 15.0, # Summer-winter variation
    'noise_std': 1.5,
    'update_frequency': '10T'  # Every 10 minutes
}

# Humidity Sensor (Relative Humidity %)
HUMIDITY_CONFIG = {
    'min_value': 30.0,
    'max_value': 100.0,
    'optimal_min': 50.0,
    'optimal_max': 70.0,
    'critical_high': 80.0, # High disease risk
    'daily_mean': 60.0,
    'daily_amplitude': 15.0,
    'noise_std': 3.0,
    'update_frequency': '10T'
}

# Light Sensor (PAR - Photosynthetically Active Radiation, µmol/m²/s)
LIGHT_CONFIG = {
    'min_value': 0.0,
    'max_value': 2000.0,
    'optimal_min': 400.0,
    'optimal_max': 1800.0,
    'peak_value': 1600.0,   # Noon on sunny day
    'night_value': 0.0,
    'cloudy_reduction': 0.6, # Cloud cover reduces light by 40%
    'noise_std': 50.0,
    'update_frequency': '10T'
}

# EC Sensor (Electrical Conductivity, dS/m - nutrient proxy)
EC_CONFIG = {
    'min_value': 0.0,
    'max_value': 5.0,
    'optimal_min': 1.5,
    'optimal_max': 3.0,
    'critical_low': 1.0,   # Nutrient deficiency
    'critical_high': 4.0,  # Salt stress
    'initial_value': 2.5,
    'depletion_rate': 0.01, # Per day (nutrient uptake)
    'fertilizer_boost': 1.0, # EC increase from fertilizer
    'noise_std': 0.1,
    'update_frequency': '1D'  # Daily
}

# Rain Gauge (mm/day)
RAINFALL_CONFIG = {
    'min_value': 0.0,
    'max_value': 100.0,
    'monthly_averages': {  # Typical rainfall by month (mm)
        1: 40, 2: 45, 3: 70, 4: 90, 5: 110, 6: 100,
        7: 90, 8: 85, 9: 75, 10: 65, 11: 55, 12: 45
    },
    'event_probability': 0.25,  # 25% chance of rain on any day
    'event_mean': 15.0,         # Average rainfall per event (mm)
    'event_std': 10.0,
    'update_frequency': '1D'
}

# ============================================================================
# Water Balance Model
# ============================================================================
WATER_BALANCE_CONFIG = {
    'field_capacity': 100.0,     # Maximum soil moisture (%)
    'wilting_point': 20.0,       # Permanent wilting point (%)
    'evapotranspiration_rate': 5.0,  # mm/day (depends on temp, crop)
    'drainage_coefficient': 0.1,  # Fraction of excess water lost per day
    'irrigation_amount': 20.0,   # mm per irrigation event
}

# ============================================================================
# AI Model Configuration
# ============================================================================

# Crop Health Classifier
CROP_HEALTH_CONFIG = {
    'model_type': 'random_forest',  # 'random_forest', 'gradient_boosting', 'neural_network'
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_SEED,
    'classes': {
        0: 'Critical',
        1: 'Poor',
        2: 'Fair',
        3: 'Good',
        4: 'Excellent'
    },
    'features': [
        'soil_moisture',
        'temperature',
        'humidity',
        'light',
        'ec',
        'days_since_planting',
        'rainfall_7d'
    ]
}

# Yield Predictor
YIELD_PREDICTOR_CONFIG = {
    'model_type': 'gradient_boosting',  # 'gradient_boosting', 'random_forest', 'linear'
    'n_estimators': 150,
    'learning_rate': 0.1,
    'max_depth': 8,
    'random_state': RANDOM_SEED,
    'features': [
        'avg_soil_moisture',
        'avg_temperature',
        'avg_ec',
        'total_rainfall',
        'stress_days',
        'optimal_days',
        'growth_period_days'
    ],
    'expected_yield_range': (5000, 12000),  # kg/ha for corn
}

# ============================================================================
# Alert System Configuration
# ============================================================================

ALERT_CONFIG = {
    'check_frequency': '1H',  # How often to check conditions
    'notification_method': 'log',  # 'log', 'email', 'sms', 'webhook'
    'alert_types': {
        'drought_stress': {
            'enabled': True,
            'condition': 'soil_moisture < 30 for 2+ days',
            'priority': 'high',
            'action': 'Irrigate immediately'
        },
        'nutrient_deficiency': {
            'enabled': True,
            'condition': 'ec < 1.0',
            'priority': 'medium',
            'action': 'Apply fertilizer'
        },
        'heat_stress': {
            'enabled': True,
            'condition': 'temperature > 35 for 3+ hours',
            'priority': 'high',
            'action': 'Increase irrigation, provide shade if possible'
        },
        'frost_warning': {
            'enabled': True,
            'condition': 'temperature < 5',
            'priority': 'high',
            'action': 'Implement frost protection measures'
        },
        'disease_risk': {
            'enabled': True,
            'condition': 'humidity > 80 and 20 < temperature < 25',
            'priority': 'medium',
            'action': 'Monitor for fungal diseases'
        },
        'overwatering': {
            'enabled': True,
            'condition': 'soil_moisture > 75 for 3+ days',
            'priority': 'low',
            'action': 'Reduce irrigation'
        }
    }
}

# ============================================================================
# Scenario Presets
# ============================================================================

SCENARIOS = {
    'optimal': {
        'description': 'Ideal growing conditions',
        'rainfall_modifier': 1.0,      # Normal rainfall
        'temperature_modifier': 0.0,   # Normal temperature
        'ec_depletion_modifier': 0.8,  # Slow nutrient depletion
        'irrigation_enabled': True,
        'fertilization_enabled': True
    },
    'drought': {
        'description': 'Drought stress scenario',
        'rainfall_modifier': 0.3,      # 70% reduction
        'temperature_modifier': 5.0,   # +5°C hotter
        'ec_depletion_modifier': 1.2,  # Faster depletion
        'irrigation_enabled': True,
        'fertilization_enabled': True
    },
    'nutrient_deficiency': {
        'description': 'Low nutrient availability',
        'rainfall_modifier': 1.0,
        'temperature_modifier': 0.0,
        'ec_depletion_modifier': 2.0,  # Rapid depletion
        'irrigation_enabled': True,
        'fertilization_enabled': False  # No fertilizer applied
    },
    'excessive_rain': {
        'description': 'Flood conditions',
        'rainfall_modifier': 2.5,      # 2.5x normal rainfall
        'temperature_modifier': -3.0,  # Cooler
        'ec_depletion_modifier': 1.5,  # Leaching
        'irrigation_enabled': False,   # No irrigation needed
        'fertilization_enabled': True
    },
    'random': {
        'description': 'Realistic variable conditions',
        'rainfall_modifier': 'random',  # Will vary
        'temperature_modifier': 'random',
        'ec_depletion_modifier': 1.0,
        'irrigation_enabled': True,
        'fertilization_enabled': True
    }
}

# ============================================================================
# Visualization Settings
# ============================================================================

VISUALIZATION_CONFIG = {
    'plot_style': 'seaborn',
    'figure_dpi': 150,
    'color_palette': 'husl',
    'sensor_colors': {
        'soil_moisture': '#1f77b4',
        'temperature': '#ff7f0e',
        'humidity': '#2ca02c',
        'light': '#d62728',
        'ec': '#9467bd',
        'rainfall': '#8c564b'
    },
    'health_colors': {
        'Critical': '#d62728',
        'Poor': '#ff7f0e',
        'Fair': '#ffdd57',
        'Good': '#7fc97f',
        'Excellent': '#2ca02c'
    },
    'dashboard_layout': 'grid',  # 'grid', 'tabs', 'single'
    'interactive': True,         # Use plotly for interactive plots
    'save_format': 'png'         # 'png', 'pdf', 'svg'
}

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_CONFIG = {
    'save_sensor_data': True,
    'save_predictions': True,
    'save_alerts': True,
    'save_plots': True,
    'create_summary_report': True,
    'sensor_data_file': os.path.join(DATA_DIR, 'sensor_data.csv'),
    'predictions_file': os.path.join(DATA_DIR, 'predictions.csv'),
    'alerts_file': os.path.join(DATA_DIR, 'alerts.csv'),
}

# ============================================================================
# Helper Functions
# ============================================================================

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    print("✓ Directories created/verified")


def print_config():
    """Print current configuration summary"""
    print("=" * 60)
    print("Smart Agriculture IoT Simulation - Configuration")
    print("=" * 60)
    print(f"Crop type: {CROP_TYPE}")
    print(f"Field area: {FIELD_AREA_HA} ha")
    print(f"Simulation period: {SIMULATION_DAYS} days")
    print(f"Start date: {SIMULATION_START_DATE.strftime('%Y-%m-%d')}")
    print(f"Planting date: {PLANTING_DATE.strftime('%Y-%m-%d')}")
    print(f"Growth period: {GROWTH_PERIOD_DAYS} days")
    print(f"Random seed: {RANDOM_SEED}")
    print("\nSensors configured:")
    print(f"  - Soil Moisture")
    print(f"  - Temperature")
    print(f"  - Humidity")
    print(f"  - Light (PAR)")
    print(f"  - EC (nutrients)")
    print(f"  - Rainfall")
    print("\nAI Models:")
    print(f"  - Crop Health: {CROP_HEALTH_CONFIG['model_type']}")
    print(f"  - Yield Predictor: {YIELD_PREDICTOR_CONFIG['model_type']}")
    print("\nAlert System:")
    print(f"  - {sum(1 for a in ALERT_CONFIG['alert_types'].values() if a['enabled'])} alert types enabled")
    print("=" * 60)


def get_scenario(scenario_name):
    """
    Get scenario configuration by name

    Args:
        scenario_name: Name of scenario ('optimal', 'drought', etc.)

    Returns:
        dict: Scenario configuration
    """
    if scenario_name not in SCENARIOS:
        print(f"Warning: Unknown scenario '{scenario_name}', using 'random'")
        scenario_name = 'random'

    return SCENARIOS[scenario_name]


if __name__ == "__main__":
    create_directories()
    print_config()

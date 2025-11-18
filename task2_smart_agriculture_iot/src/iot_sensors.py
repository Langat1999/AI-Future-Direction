"""
IoT Sensor Simulation Classes
Simulates realistic agricultural sensors with physics-based models
"""

import numpy as np
from datetime import datetime, timedelta
import config


class BaseSensor:
    """Base class for all sensors"""

    def __init__(self, name, config_dict, random_state=None):
        """
        Initialize base sensor

        Args:
            name: Sensor name
            config_dict: Configuration dictionary
            random_state: Random state for reproducibility
        """
        self.name = name
        self.config = config_dict
        self.rng = np.random.RandomState(random_state)
        self.current_value = config_dict.get('initial_value',
                                             (config_dict['min_value'] + config_dict['max_value']) / 2)
        self.history = []

    def add_noise(self, value):
        """Add measurement noise"""
        noise_std = self.config.get('noise_std', 0)
        noise = self.rng.normal(0, noise_std)
        return value + noise

    def clip_value(self, value):
        """Clip value to sensor range"""
        return np.clip(value, self.config['min_value'], self.config['max_value'])

    def read(self):
        """Read current sensor value"""
        return self.current_value

    def update(self, timestamp, **kwargs):
        """Update sensor reading (to be implemented by subclasses)"""
        raise NotImplementedError


class SoilMoistureSensor(BaseSensor):
    """
    Soil moisture sensor with water balance model
    Simulates realistic soil moisture dynamics
    """

    def __init__(self, random_state=None):
        super().__init__('Soil Moisture', config.SOIL_MOISTURE_CONFIG, random_state)
        self.water_balance_config = config.WATER_BALANCE_CONFIG

    def update(self, timestamp, rainfall=0, irrigation=0, temperature=25, **kwargs):
        """
        Update soil moisture using water balance

        Args:
            timestamp: Current timestamp
            rainfall: Rainfall amount (mm)
            irrigation: Irrigation amount (mm)
            temperature: Air temperature (affects ET)

        Returns:
            float: Updated soil moisture (%)
        """
        # Calculate evapotranspiration (temperature-dependent)
        base_et = self.water_balance_config['evapotranspiration_rate']
        temp_factor = 1.0 + (temperature - 25) * 0.02  # 2% change per degree
        et = base_et * temp_factor

        # Water inputs
        water_in = rainfall + irrigation

        # Water outputs
        water_out = et

        # Drainage (only if above field capacity)
        field_capacity = self.water_balance_config['field_capacity']
        if self.current_value > field_capacity * 0.8:  # 80% of field capacity
            excess = self.current_value - field_capacity * 0.8
            drainage = excess * self.water_balance_config['drainage_coefficient']
            water_out += drainage

        # Update soil moisture (simple conversion: mm to %)
        # Assuming 1mm water ≈ 1% volumetric water content (simplified)
        delta_moisture = water_in - water_out
        new_value = self.current_value + delta_moisture

        # Add noise and clip
        new_value = self.add_noise(new_value)
        new_value = self.clip_value(new_value)

        self.current_value = new_value
        self.history.append({
            'timestamp': timestamp,
            'value': new_value,
            'rainfall': rainfall,
            'irrigation': irrigation,
            'et': et
        })

        return new_value


class TemperatureSensor(BaseSensor):
    """
    Air temperature sensor with daily and seasonal cycles
    """

    def __init__(self, random_state=None):
        super().__init__('Temperature', config.TEMPERATURE_CONFIG, random_state)
        self.start_date = config.SIMULATION_START_DATE

    def update(self, timestamp, **kwargs):
        """
        Update temperature with sinusoidal daily and seasonal patterns

        Args:
            timestamp: Current timestamp

        Returns:
            float: Temperature in Celsius
        """
        # Days since simulation start
        days_since_start = (timestamp - self.start_date).total_seconds() / 86400

        # Seasonal component (annual cycle)
        seasonal = self.config['seasonal_amplitude'] * np.sin(2 * np.pi * days_since_start / 365)

        # Daily component (24-hour cycle)
        hours = timestamp.hour + timestamp.minute / 60
        daily = self.config['daily_amplitude'] * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at 2pm

        # Base temperature
        base_temp = self.config['daily_mean']

        # Combine components
        temperature = base_temp + seasonal + daily

        # Add noise
        temperature = self.add_noise(temperature)
        temperature = self.clip_value(temperature)

        self.current_value = temperature
        self.history.append({
            'timestamp': timestamp,
            'value': temperature
        })

        return temperature


class HumiditySensor(BaseSensor):
    """
    Relative humidity sensor (inversely correlated with temperature)
    """

    def __init__(self, random_state=None):
        super().__init__('Humidity', config.HUMIDITY_CONFIG, random_state)

    def update(self, timestamp, temperature=None, **kwargs):
        """
        Update humidity (inversely related to temperature)

        Args:
            timestamp: Current timestamp
            temperature: Current air temperature

        Returns:
            float: Relative humidity (%)
        """
        # Base humidity
        base_humidity = self.config['daily_mean']

        # Temperature effect (inverse relationship)
        if temperature is not None:
            temp_deviation = temperature - 25  # Reference temperature
            humidity_change = -temp_deviation * 1.5  # 1.5% RH change per degree
        else:
            humidity_change = 0

        # Daily cycle (opposite to temperature)
        hours = timestamp.hour + timestamp.minute / 60
        daily = self.config['daily_amplitude'] * np.sin(2 * np.pi * (hours - 18) / 24)  # Peak at 6am

        # Combine
        humidity = base_humidity + humidity_change + daily

        # Add noise
        humidity = self.add_noise(humidity)
        humidity = self.clip_value(humidity)

        self.current_value = humidity
        self.history.append({
            'timestamp': timestamp,
            'value': humidity,
            'temperature': temperature
        })

        return humidity


class LightSensor(BaseSensor):
    """
    PAR (Photosynthetically Active Radiation) sensor
    Simulates day/night cycle and cloud cover
    """

    def __init__(self, random_state=None):
        super().__init__('Light', config.LIGHT_CONFIG, random_state)

    def update(self, timestamp, cloud_cover=0.3, **kwargs):
        """
        Update light level (PAR)

        Args:
            timestamp: Current timestamp
            cloud_cover: Cloud cover fraction (0-1)

        Returns:
            float: PAR in µmol/m²/s
        """
        hours = timestamp.hour + timestamp.minute / 60

        # Day/night cycle (gaussian-like curve centered at noon)
        if 6 <= hours <= 18:  # Daylight hours
            # Parabolic curve peaking at noon
            time_factor = 1 - ((hours - 12) / 6) ** 2
            light = self.config['peak_value'] * max(0, time_factor)
        else:
            light = self.config['night_value']

        # Cloud cover effect
        cloud_reduction = 1 - (cloud_cover * (1 - self.config['cloudy_reduction']))
        light = light * cloud_reduction

        # Add noise
        light = self.add_noise(light)
        light = self.clip_value(light)

        self.current_value = light
        self.history.append({
            'timestamp': timestamp,
            'value': light,
            'cloud_cover': cloud_cover
        })

        return light


class ECSensor(BaseSensor):
    """
    Electrical Conductivity sensor (nutrient proxy)
    Depletes over time from plant uptake, replenished by fertilizer
    """

    def __init__(self, random_state=None):
        super().__init__('EC', config.EC_CONFIG, random_state)

    def update(self, timestamp, fertilizer_applied=False, rainfall=0, **kwargs):
        """
        Update EC level

        Args:
            timestamp: Current timestamp
            fertilizer_applied: Whether fertilizer was applied
            rainfall: Rainfall amount (can cause leaching)

        Returns:
            float: EC in dS/m
        """
        # Natural depletion (nutrient uptake)
        depletion = self.config['depletion_rate']

        # Rainfall leaching (heavy rain reduces EC)
        if rainfall > 20:  # Heavy rain threshold
            leaching = (rainfall - 20) * 0.01
        else:
            leaching = 0

        # Update EC
        new_value = self.current_value - depletion - leaching

        # Fertilizer application
        if fertilizer_applied:
            new_value += self.config['fertilizer_boost']

        # Add noise
        new_value = self.add_noise(new_value)
        new_value = self.clip_value(new_value)

        self.current_value = new_value
        self.history.append({
            'timestamp': timestamp,
            'value': new_value,
            'fertilizer_applied': fertilizer_applied,
            'rainfall': rainfall
        })

        return new_value


class RainGauge(BaseSensor):
    """
    Rain gauge with stochastic rainfall events
    """

    def __init__(self, random_state=None):
        super().__init__('Rainfall', config.RAINFALL_CONFIG, random_state)

    def update(self, timestamp, scenario_modifier=1.0, **kwargs):
        """
        Update rainfall (stochastic events)

        Args:
            timestamp: Current timestamp
            scenario_modifier: Multiplier for rainfall amount

        Returns:
            float: Rainfall in mm
        """
        # Get monthly average
        month = timestamp.month
        monthly_avg = self.config['monthly_averages'].get(month, 50)

        # Stochastic event
        if self.rng.random() < self.config['event_probability']:
            # Rain event occurs
            rainfall = self.rng.normal(self.config['event_mean'],
                                      self.config['event_std'])
            rainfall = max(0, rainfall)  # No negative rain

            # Apply scenario modifier
            rainfall *= scenario_modifier
        else:
            rainfall = 0

        rainfall = self.clip_value(rainfall)

        self.current_value = rainfall
        self.history.append({
            'timestamp': timestamp,
            'value': rainfall
        })

        return rainfall


class SensorNetwork:
    """
    Network of all agricultural sensors
    Coordinates sensor readings and dependencies
    """

    def __init__(self, random_state=None):
        """Initialize all sensors"""
        self.random_state = random_state

        # Create sensors
        self.soil_moisture = SoilMoistureSensor(random_state)
        self.temperature = TemperatureSensor(random_state)
        self.humidity = HumiditySensor(random_state)
        self.light = LightSensor(random_state)
        self.ec = ECSensor(random_state)
        self.rainfall = RainGauge(random_state)

        self.sensors = {
            'soil_moisture': self.soil_moisture,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'light': self.light,
            'ec': self.ec,
            'rainfall': self.rainfall
        }

    def read_all(self, timestamp, irrigation=0, fertilizer=False,
                 cloud_cover=0.3, scenario_modifier=1.0):
        """
        Read all sensors (coordinated update)

        Args:
            timestamp: Current timestamp
            irrigation: Irrigation amount (mm)
            fertilizer: Whether fertilizer was applied
            cloud_cover: Cloud cover fraction
            scenario_modifier: Rainfall modifier for scenarios

        Returns:
            dict: All sensor readings
        """
        # Update in dependency order

        # 1. Independent sensors
        rainfall = self.rainfall.update(timestamp, scenario_modifier=scenario_modifier)
        temperature = self.temperature.update(timestamp)
        light = self.light.update(timestamp, cloud_cover=cloud_cover)

        # 2. Dependent sensors
        humidity = self.humidity.update(timestamp, temperature=temperature)
        ec = self.ec.update(timestamp, fertilizer_applied=fertilizer, rainfall=rainfall)
        soil_moisture = self.soil_moisture.update(timestamp,
                                                   rainfall=rainfall,
                                                   irrigation=irrigation,
                                                   temperature=temperature)

        readings = {
            'timestamp': timestamp,
            'soil_moisture': soil_moisture,
            'temperature': temperature,
            'humidity': humidity,
            'light': light,
            'ec': ec,
            'rainfall': rainfall
        }

        return readings

    def get_current_readings(self):
        """Get current values from all sensors"""
        return {
            'soil_moisture': self.soil_moisture.read(),
            'temperature': self.temperature.read(),
            'humidity': self.humidity.read(),
            'light': self.light.read(),
            'ec': self.ec.read(),
            'rainfall': self.rainfall.read()
        }

    def reset(self):
        """Reset all sensors to initial state"""
        for sensor in self.sensors.values():
            sensor.current_value = sensor.config.get('initial_value',
                                                     (sensor.config['min_value'] + sensor.config['max_value']) / 2)
            sensor.history = []


if __name__ == "__main__":
    # Test sensor network
    print("Testing Sensor Network...")

    network = SensorNetwork(random_state=42)

    # Simulate 7 days
    start_time = datetime(2025, 6, 1)

    print("\nSimulating 7 days of sensor readings:")
    print("=" * 80)
    print(f"{'Day':<5} {'Time':<8} {'Soil%':<8} {'Temp°C':<8} {'Humid%':<8} {'PAR':<8} {'EC':<8} {'Rain':<8}")
    print("=" * 80)

    for day in range(7):
        for hour in [0, 6, 12, 18]:  # 4 readings per day
            timestamp = start_time + timedelta(days=day, hours=hour)

            # Simulate some irrigation and fertilizer
            irrigation = 20 if day == 2 and hour == 6 else 0
            fertilizer = day == 4 and hour == 0

            readings = network.read_all(timestamp,
                                       irrigation=irrigation,
                                       fertilizer=fertilizer,
                                       cloud_cover=0.2)

            print(f"{day+1:<5} {hour:02d}:00  "
                  f"{readings['soil_moisture']:>6.1f}  "
                  f"{readings['temperature']:>6.1f}  "
                  f"{readings['humidity']:>6.1f}  "
                  f"{readings['light']:>6.0f}  "
                  f"{readings['ec']:>6.2f}  "
                  f"{readings['rainfall']:>6.1f}")

    print("=" * 80)
    print("\n✓ Sensor network test complete")

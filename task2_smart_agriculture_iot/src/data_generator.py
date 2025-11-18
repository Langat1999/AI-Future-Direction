"""
Synthetic Agricultural Data Generator
Generates realistic time-series sensor data for simulation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import config
from iot_sensors import SensorNetwork


class AgricultureDataGenerator:
    """
    Generates synthetic agricultural sensor data with realistic patterns
    """

    def __init__(self, scenario='optimal', random_state=None):
        """
        Initialize data generator

        Args:
            scenario: Scenario name ('optimal', 'drought', etc.)
            random_state: Random state for reproducibility
        """
        self.scenario = config.get_scenario(scenario)
        self.scenario_name = scenario
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Initialize sensor network
        self.sensor_network = SensorNetwork(random_state)

        # Management actions log
        self.irrigation_events = []
        self.fertilizer_events = []

    def generate(self, start_date=None, days=365, save_to_file=True):
        """
        Generate time-series sensor data

        Args:
            start_date: Start date for simulation
            days: Number of days to simulate
            save_to_file: Whether to save data to CSV

        Returns:
            pd.DataFrame: Generated sensor data
        """
        start_date = start_date or config.SIMULATION_START_DATE

        print(f"\nGenerating {days} days of sensor data ({self.scenario_name} scenario)...")

        # Determine reading frequency
        if config.HOURLY_READINGS:
            freq = '1H'
            total_readings = days * 24
        else:
            freq = '1D'
            total_readings = days

        # Create timestamp index
        timestamps = pd.date_range(start=start_date, periods=total_readings, freq=freq)

        # Initialize data storage
        data_records = []

        # Scenario modifiers
        rainfall_mod = self.scenario['rainfall_modifier']
        temp_mod = self.scenario['temperature_modifier']
        ec_depletion_mod = self.scenario['ec_depletion_modifier']

        # Adjust sensor configs for scenario
        if temp_mod != 0:
            self.sensor_network.temperature.config['daily_mean'] += temp_mod

        if ec_depletion_mod != 1.0:
            self.sensor_network.ec.config['depletion_rate'] *= ec_depletion_mod

        # Simulate each timestamp
        for idx, timestamp in enumerate(tqdm(timestamps, desc="Simulating")):

            # Determine management actions
            irrigation = self._decide_irrigation(timestamp)
            fertilizer = self._decide_fertilizer(timestamp)

            # Cloud cover (random)
            cloud_cover = self.rng.uniform(0.1, 0.7)

            # Get sensor readings
            if isinstance(rainfall_mod, str) and rainfall_mod == 'random':
                rain_mod = self.rng.uniform(0.5, 1.5)
            else:
                rain_mod = rainfall_mod

            readings = self.sensor_network.read_all(
                timestamp=timestamp,
                irrigation=irrigation,
                fertilizer=fertilizer,
                cloud_cover=cloud_cover,
                scenario_modifier=rain_mod
            )

            # Calculate derived features
            days_since_planting = max(0, (timestamp - config.PLANTING_DATE).days)

            # Store reading
            record = {
                'timestamp': timestamp,
                'soil_moisture': readings['soil_moisture'],
                'temperature': readings['temperature'],
                'humidity': readings['humidity'],
                'light': readings['light'],
                'ec': readings['ec'],
                'rainfall': readings['rainfall'],
                'irrigation': irrigation,
                'fertilizer_applied': fertilizer,
                'days_since_planting': days_since_planting,
                'cloud_cover': cloud_cover
            }

            data_records.append(record)

        # Create DataFrame
        df = pd.DataFrame(data_records)

        # Add derived features
        df = self._add_derived_features(df)

        # Save to file
        if save_to_file:
            output_path = config.OUTPUT_CONFIG['sensor_data_file']
            df.to_csv(output_path, index=False)
            print(f"✓ Data saved to {output_path}")

        print(f"✓ Generated {len(df)} sensor readings")

        return df

    def _decide_irrigation(self, timestamp):
        """
        Decide whether to irrigate (simple rule-based)

        Args:
            timestamp: Current timestamp

        Returns:
            float: Irrigation amount (mm), 0 if no irrigation
        """
        if not self.scenario.get('irrigation_enabled', True):
            return 0

        current_moisture = self.sensor_network.soil_moisture.read()

        # Irrigate if soil moisture falls below threshold
        if current_moisture < 35:
            amount = config.WATER_BALANCE_CONFIG['irrigation_amount']
            self.irrigation_events.append({
                'timestamp': timestamp,
                'amount': amount,
                'reason': f'Low soil moisture ({current_moisture:.1f}%)'
            })
            return amount

        return 0

    def _decide_fertilizer(self, timestamp):
        """
        Decide whether to apply fertilizer

        Args:
            timestamp: Current timestamp

        Returns:
            bool: Whether to apply fertilizer
        """
        if not self.scenario.get('fertilization_enabled', True):
            return False

        current_ec = self.sensor_network.ec.read()
        days_since_planting = (timestamp - config.PLANTING_DATE).days

        # Fertilize at planting and every 30 days if EC is low
        if days_since_planting == 0:
            apply = True
            reason = "Planting time"
        elif days_since_planting % 30 == 0 and current_ec < 2.0:
            apply = True
            reason = f"Scheduled + low EC ({current_ec:.2f})"
        elif current_ec < 1.0:
            apply = True
            reason = f"Critical EC ({current_ec:.2f})"
        else:
            apply = False
            reason = None

        if apply:
            self.fertilizer_events.append({
                'timestamp': timestamp,
                'reason': reason
            })

        return apply

    def _add_derived_features(self, df):
        """
        Add rolling statistics and derived features

        Args:
            df: DataFrame with sensor data

        Returns:
            pd.DataFrame: DataFrame with derived features
        """
        # Rolling averages (7-day window)
        df['soil_moisture_7d'] = df['soil_moisture'].rolling(window=168, min_periods=1).mean()
        df['temperature_7d'] = df['temperature'].rolling(window=168, min_periods=1).mean()
        df['rainfall_7d'] = df['rainfall'].rolling(window=168, min_periods=1).sum()

        # Stress indicators
        df['drought_stress'] = (df['soil_moisture'] < 30).astype(int)
        df['heat_stress'] = (df['temperature'] > 35).astype(int)
        df['nutrient_stress'] = (df['ec'] < 1.0).astype(int)

        # Vapor Pressure Deficit (VPD) - simplified
        # VPD = (1 - RH/100) * SVP(T)
        # Simplified SVP calculation
        svp = 0.6108 * np.exp((17.27 * df['temperature']) / (df['temperature'] + 237.3))
        df['vpd'] = (1 - df['humidity'] / 100) * svp

        # Growing Degree Days (GDD) - base 10°C
        df['gdd'] = np.maximum(0, df['temperature'] - 10)
        df['gdd_cumulative'] = df['gdd'].cumsum()

        return df

    def get_summary_statistics(self, df):
        """
        Calculate summary statistics for the generated data

        Args:
            df: DataFrame with sensor data

        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_days': len(df) / 24 if config.HOURLY_READINGS else len(df),
            'total_readings': len(df),
            'sensor_stats': {},
            'stress_days': {},
            'management_actions': {
                'irrigation_events': len(self.irrigation_events),
                'total_irrigation_mm': sum(e['amount'] for e in self.irrigation_events),
                'fertilizer_events': len(self.fertilizer_events)
            }
        }

        # Sensor statistics
        for sensor in ['soil_moisture', 'temperature', 'humidity', 'light', 'ec', 'rainfall']:
            summary['sensor_stats'][sensor] = {
                'mean': df[sensor].mean(),
                'std': df[sensor].std(),
                'min': df[sensor].min(),
                'max': df[sensor].max(),
                'median': df[sensor].median()
            }

        # Stress days
        if config.HOURLY_READINGS:
            # Count days with stress (at least 6 hours of stress)
            daily = df.set_index('timestamp').resample('1D').mean()
            summary['stress_days']['drought'] = (daily['drought_stress'] > 0.25).sum()
            summary['stress_days']['heat'] = (daily['heat_stress'] > 0.25).sum()
            summary['stress_days']['nutrient'] = (daily['nutrient_stress'] > 0.5).sum()
        else:
            summary['stress_days']['drought'] = df['drought_stress'].sum()
            summary['stress_days']['heat'] = df['heat_stress'].sum()
            summary['stress_days']['nutrient'] = df['nutrient_stress'].sum()

        return summary


def generate_multiple_seasons(num_seasons=3, scenario='random', random_state=None):
    """
    Generate multiple seasons of data for model training

    Args:
        num_seasons: Number of seasons to generate
        scenario: Scenario type
        random_state: Random seed

    Returns:
        pd.DataFrame: Multi-season dataset
    """
    all_data = []

    for season in range(num_seasons):
        print(f"\n=== Generating Season {season + 1}/{num_seasons} ===")

        # Different start date for each season
        start_date = config.SIMULATION_START_DATE + timedelta(days=season * 365)

        # Create generator with different seed
        seed = random_state + season if random_state else None
        generator = AgricultureDataGenerator(scenario=scenario, random_state=seed)

        # Generate data
        df = generator.generate(start_date=start_date, days=365, save_to_file=False)
        df['season'] = season + 1

        all_data.append(df)

    # Combine all seasons
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save combined data
    output_path = config.OUTPUT_CONFIG['sensor_data_file'].replace('.csv', '_multiseason.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Combined data saved to {output_path}")

    return combined_df


if __name__ == "__main__":
    # Test data generation
    print("=" * 60)
    print("Agriculture Data Generator - Test")
    print("=" * 60)

    # Test single season
    generator = AgricultureDataGenerator(scenario='optimal', random_state=42)
    df = generator.generate(days=30, save_to_file=False)  # 30 days for testing

    print("\nGenerated Data Sample:")
    print(df.head(10))

    print("\nSummary Statistics:")
    summary = generator.get_summary_statistics(df)

    print(f"\nSensor Averages:")
    for sensor, stats in summary['sensor_stats'].items():
        print(f"  {sensor:15s}: {stats['mean']:8.2f} (±{stats['std']:.2f})")

    print(f"\nStress Days:")
    for stress_type, days in summary['stress_days'].items():
        print(f"  {stress_type:15s}: {days} days")

    print(f"\nManagement:")
    print(f"  Irrigation events: {summary['management_actions']['irrigation_events']}")
    print(f"  Total irrigation: {summary['management_actions']['total_irrigation_mm']:.0f} mm")
    print(f"  Fertilizer events: {summary['management_actions']['fertilizer_events']}")

    print("\n✓ Data generation test complete")

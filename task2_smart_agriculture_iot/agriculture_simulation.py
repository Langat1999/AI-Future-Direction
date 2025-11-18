"""
Smart Agriculture IoT Simulation - Main Script
Comprehensive simulation integrating sensors, AI, alerts, and visualization
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import config
from data_generator import AgricultureDataGenerator, generate_multiple_seasons
from ai_predictor import CropHealthPredictor, YieldPredictor
from alert_system import AlertSystem, IrrigationDecisionSystem, FertilizerDecisionSystem
from visualization import AgricultureDashboard, create_comprehensive_report


class SmartAgricultureSimulation:
    """
    Main simulation class integrating all components
    """

    def __init__(self, scenario='optimal', days=365, random_state=None):
        """
        Initialize simulation

        Args:
            scenario: Scenario type ('optimal', 'drought', etc.)
            days: Number of days to simulate
            random_state: Random seed
        """
        self.scenario = scenario
        self.days = days
        self.random_state = random_state

        # Components
        self.data_generator = None
        self.sensor_data = None
        self.crop_health_predictor = CropHealthPredictor(random_state)
        self.yield_predictor = YieldPredictor(random_state)
        self.alert_system = AlertSystem()
        self.irrigation_system = IrrigationDecisionSystem()
        self.fertilizer_system = FertilizerDecisionSystem()

        # Results
        self.predictions = None
        self.alerts = []
        self.summary = {}

    def run(self, train_models=True, generate_visualizations=True):
        """
        Run complete simulation

        Args:
            train_models: Whether to train AI models
            generate_visualizations: Whether to create plots

        Returns:
            dict: Simulation summary
        """
        print("=" * 60)
        print("Smart Agriculture IoT Simulation")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  Scenario: {self.scenario}")
        print(f"  Duration: {self.days} days")
        print(f"  Crop: {config.CROP_TYPE}")
        print(f"  Field area: {config.FIELD_AREA_HA} ha")
        print(f"  Random seed: {self.random_state}")

        # Step 1: Generate sensor data
        print("\n" + "-" * 60)
        print("Step 1: Generating Sensor Data")
        print("-" * 60)

        self.data_generator = AgricultureDataGenerator(
            scenario=self.scenario,
            random_state=self.random_state
        )

        self.sensor_data = self.data_generator.generate(
            days=self.days,
            save_to_file=True
        )

        # Print data summary
        summary_stats = self.data_generator.get_summary_statistics(self.sensor_data)
        self._print_data_summary(summary_stats)

        # Step 2: Train AI models
        if train_models:
            print("\n" + "-" * 60)
            print("Step 2: Training AI Models")
            print("-" * 60)

            # Train crop health classifier
            health_metrics = self.crop_health_predictor.train(self.sensor_data, save_model=True)

            # For yield prediction, we need multiple seasons
            print("\nGenerating additional seasons for yield prediction...")
            multi_season_data = generate_multiple_seasons(
                num_seasons=5,
                scenario=self.scenario,
                random_state=self.random_state
            )

            yield_metrics = self.yield_predictor.train(multi_season_data, save_model=True)

            self.summary['model_performance'] = {
                'crop_health_accuracy': health_metrics['accuracy'],
                'yield_r2_score': yield_metrics['r2_score'],
                'yield_rmse': yield_metrics['rmse']
            }

        # Step 3: Run predictions
        print("\n" + "-" * 60)
        print("Step 3: Running Predictions and Alerts")
        print("-" * 60)

        self._run_predictions_and_alerts()

        # Step 4: Generate summary
        print("\n" + "-" * 60)
        print("Step 4: Generating Summary")
        print("-" * 60)

        self._generate_summary()
        self._print_summary()

        # Step 5: Create visualizations
        if generate_visualizations:
            print("\n" + "-" * 60)
            print("Step 5: Creating Visualizations")
            print("-" * 60)

            self._create_visualizations()

        # Save summary
        self._save_summary()

        print("\n" + "=" * 60)
        print("Simulation Complete!")
        print("=" * 60)
        print(f"\nResults saved to:")
        print(f"  Data: {config.OUTPUT_CONFIG['sensor_data_file']}")
        print(f"  Predictions: {config.OUTPUT_CONFIG['predictions_file']}")
        print(f"  Alerts: {config.OUTPUT_CONFIG['alerts_file']}")
        print(f"  Visualizations: {config.VISUALIZATION_DIR}")
        print("=" * 60)

        return self.summary

    def _run_predictions_and_alerts(self):
        """Run crop health predictions and check alerts"""
        predictions_list = []

        # Process each data point
        for idx, row in self.sensor_data.iterrows():
            # Prepare features for crop health prediction
            features = {
                'soil_moisture': row['soil_moisture'],
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'light': row['light'],
                'ec': row['ec'],
                'days_since_planting': row['days_since_planting'],
                'rainfall_7d': row.get('rainfall_7d', 0)
            }

            # Predict crop health
            health_result = self.crop_health_predictor.predict(features)

            # Check for alerts
            alerts = self.alert_system.check_conditions(row, timestamp=row['timestamp'])
            self.alerts.extend(alerts)

            # Store prediction
            predictions_list.append({
                'timestamp': row['timestamp'],
                'health_class': health_result['health_class'],
                'health_label': health_result['health_label'],
                'health_confidence': health_result['confidence'],
                'num_alerts': len(alerts)
            })

        # Create predictions DataFrame
        self.predictions = pd.DataFrame(predictions_list)

        # Save predictions
        self.predictions.to_csv(config.OUTPUT_CONFIG['predictions_file'], index=False)
        print(f"✓ Predictions saved ({len(self.predictions)} records)")

        # Save alerts
        if self.alerts:
            self.alert_system.save_alerts()
            print(f"✓ Alerts saved ({len(self.alerts)} alerts)")

    def _generate_summary(self):
        """Generate simulation summary statistics"""
        # Alert summary
        alert_summary = self.alert_system.get_alert_summary()

        # Health summary
        health_dist = self.predictions['health_label'].value_counts()
        avg_health = self.predictions['health_class'].mean()

        # Sensor summary
        sensor_summary = {
            'avg_soil_moisture': self.sensor_data['soil_moisture'].mean(),
            'avg_temperature': self.sensor_data['temperature'].mean(),
            'avg_ec': self.sensor_data['ec'].mean(),
            'total_rainfall': self.sensor_data['rainfall'].sum(),
            'total_irrigation': self.sensor_data['irrigation'].sum()
        }

        # Stress days
        stress_summary = {
            'drought_days': self.sensor_data['drought_stress'].sum(),
            'heat_days': self.sensor_data['heat_stress'].sum(),
            'nutrient_days': self.sensor_data['nutrient_stress'].sum()
        }

        # Yield prediction (for the season)
        season_features = {
            'avg_soil_moisture': sensor_summary['avg_soil_moisture'],
            'avg_temperature': sensor_summary['avg_temperature'],
            'avg_ec': sensor_summary['avg_ec'],
            'total_rainfall': sensor_summary['total_rainfall'],
            'stress_days': sum(stress_summary.values()),
            'optimal_days': self.days - sum(stress_summary.values()),
            'growth_period_days': config.GROWTH_PERIOD_DAYS
        }

        predicted_yield = self.yield_predictor.predict(season_features)

        self.summary = {
            'scenario': self.scenario,
            'duration_days': self.days,
            'alerts': alert_summary,
            'health_distribution': health_dist.to_dict(),
            'avg_health_score': avg_health,
            'sensors': sensor_summary,
            'stress_days': stress_summary,
            'predicted_yield_kg_ha': predicted_yield,
            'management_actions': {
                'irrigation_events': len(self.data_generator.irrigation_events),
                'total_irrigation_mm': sensor_summary['total_irrigation'],
                'fertilizer_events': len(self.data_generator.fertilizer_events)
            }
        }

    def _print_data_summary(self, stats):
        """Print data generation summary"""
        print(f"\nData Generation Summary:")
        print(f"  Total readings: {stats['total_readings']}")
        print(f"  Duration: {stats['total_days']:.0f} days")

        print(f"\nSensor Averages:")
        for sensor, sensor_stats in stats['sensor_stats'].items():
            print(f"  {sensor:15s}: {sensor_stats['mean']:8.2f} (±{sensor_stats['std']:.2f})")

        print(f"\nStress Days:")
        for stress_type, days in stats['stress_days'].items():
            print(f"  {stress_type:15s}: {days} days")

        print(f"\nManagement Actions:")
        mgmt = stats['management_actions']
        print(f"  Irrigation events: {mgmt['irrigation_events']} (total: {mgmt['total_irrigation_mm']:.0f} mm)")
        print(f"  Fertilizer events: {mgmt['fertilizer_events']}")

    def _print_summary(self):
        """Print simulation summary"""
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)

        print(f"\nAlerts Generated: {self.summary['alerts']['total']}")
        if self.summary['alerts']['total'] > 0:
            print("  By Type:")
            for alert_type, count in self.summary['alerts']['by_type'].items():
                print(f"    {alert_type:20s}: {count}")

        print(f"\nCrop Health Distribution:")
        for health, count in self.summary['health_distribution'].items():
            print(f"  {health:15s}: {count}")

        print(f"\nAverage Crop Health: {self.summary['avg_health_score']:.2f}/4.0 "
              f"({config.CROP_HEALTH_CONFIG['classes'][int(self.summary['avg_health_score'])]})")

        print(f"\nPredicted Yield: {self.summary['predicted_yield_kg_ha']:.0f} kg/ha")

        print(f"\nStress Days:")
        for stress_type, days in self.summary['stress_days'].items():
            print(f"  {stress_type:15s}: {days} days")

        print(f"\nManagement Summary:")
        mgmt = self.summary['management_actions']
        print(f"  Irrigation events: {mgmt['irrigation_events']} (total: {mgmt['total_irrigation_mm']:.0f} mm)")
        print(f"  Fertilizer applications: {mgmt['fertilizer_events']}")

        # Recommendations
        print(f"\nRecommendations:")
        self._generate_recommendations()

    def _generate_recommendations(self):
        """Generate management recommendations"""
        avg_moisture = self.summary['sensors']['avg_soil_moisture']
        drought_days = self.summary['stress_days']['drought_days']
        nutrient_days = self.summary['stress_days']['nutrient_days']
        avg_health = self.summary['avg_health_score']

        if avg_health >= 3.5:
            print("  ✓ Excellent management, maintain current practices")
        elif avg_health >= 2.5:
            print("  ✓ Good management overall")

        if drought_days > 30:
            print("  ⚠ Consider installing irrigation automation or increasing frequency")

        if nutrient_days > 20:
            print("  ⚠ Increase fertilizer application frequency")
            print("  ⚠ Consider soil testing for precise nutrient management")

        if avg_moisture < 40:
            print("  ⚠ Average soil moisture low - review irrigation schedule")

        if self.summary['sensors']['total_rainfall'] < 400:
            print("  ⚠ Low seasonal rainfall - supplemental irrigation critical")

    def _create_visualizations(self):
        """Create all visualizations"""
        create_comprehensive_report(
            sensor_data=self.sensor_data,
            predictions=self.predictions,
            alerts=self.alerts,
            save_dir=config.VISUALIZATION_DIR
        )

    def _save_summary(self):
        """Save summary to file"""
        import json

        summary_file = os.path.join(config.RESULTS_DIR, 'simulation_summary.json')

        # Convert non-serializable types
        summary_serializable = {}
        for key, value in self.summary.items():
            if isinstance(value, pd.Series):
                summary_serializable[key] = value.to_dict()
            elif isinstance(value, dict):
                summary_serializable[key] = {k: (v.to_dict() if isinstance(v, pd.Series) else v)
                                             for k, v in value.items()}
            else:
                summary_serializable[key] = value

        with open(summary_file, 'w') as f:
            json.dump(summary_serializable, f, indent=2)

        print(f"✓ Summary saved to {summary_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Agriculture IoT Simulation')

    parser.add_argument('--scenario', type=str, default='optimal',
                        choices=['optimal', 'drought', 'nutrient_deficiency',
                                'excessive_rain', 'random'],
                        help='Simulation scenario')

    parser.add_argument('--days', type=int, default=365,
                        help='Number of days to simulate')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--no-train', action='store_true',
                        help='Skip model training (use existing models)')

    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # Update output directory if specified
    if args.output_dir:
        config.RESULTS_DIR = args.output_dir
        config.VISUALIZATION_DIR = os.path.join(args.output_dir, 'visualizations')
        config.create_directories()

    # Create and run simulation
    simulation = SmartAgricultureSimulation(
        scenario=args.scenario,
        days=args.days,
        random_state=args.seed
    )

    simulation.run(
        train_models=not args.no_train,
        generate_visualizations=not args.no_viz
    )


if __name__ == "__main__":
    # Ensure directories exist
    config.create_directories()

    main()

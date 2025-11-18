"""
Visualization Dashboard for Smart Agriculture
Interactive plots and dashboards for sensor data, predictions, and alerts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AgricultureDashboard:
    """
    Comprehensive visualization dashboard for agriculture IoT data
    """

    def __init__(self, sensor_data=None, predictions=None, alerts=None):
        """
        Initialize dashboard

        Args:
            sensor_data: DataFrame with sensor readings
            predictions: DataFrame with predictions
            alerts: List of Alert objects or DataFrame
        """
        self.sensor_data = sensor_data
        self.predictions = predictions
        self.alerts = alerts
        self.viz_config = config.VISUALIZATION_CONFIG

    def plot_sensor_timeline(self, sensors=None, days=None, save_path=None):
        """
        Plot time-series for multiple sensors

        Args:
            sensors: List of sensor names (default: all)
            days: Number of recent days to plot (default: all)
            save_path: Path to save figure
        """
        if self.sensor_data is None:
            print("No sensor data available")
            return

        sensors = sensors or ['soil_moisture', 'temperature', 'humidity', 'ec']
        df = self.sensor_data.copy()

        # Filter to recent days if specified
        if days:
            df = df.tail(days * 24 if 'hour' in str(df.index.freq) else days)

        # Create subplots
        fig, axes = plt.subplots(len(sensors), 1, figsize=(14, 3 * len(sensors)))
        if len(sensors) == 1:
            axes = [axes]

        for idx, sensor in enumerate(sensors):
            ax = axes[idx]

            # Plot sensor data
            ax.plot(df['timestamp'], df[sensor],
                   color=self.viz_config['sensor_colors'].get(sensor, 'blue'),
                   linewidth=1.5, label=sensor.replace('_', ' ').title())

            # Add optimal range if available
            sensor_config = None
            if sensor == 'soil_moisture':
                sensor_config = config.SOIL_MOISTURE_CONFIG
            elif sensor == 'temperature':
                sensor_config = config.TEMPERATURE_CONFIG
            elif sensor == 'humidity':
                sensor_config = config.HUMIDITY_CONFIG
            elif sensor == 'ec':
                sensor_config = config.EC_CONFIG

            if sensor_config and 'optimal_min' in sensor_config:
                ax.axhspan(sensor_config['optimal_min'], sensor_config['optimal_max'],
                          alpha=0.2, color='green', label='Optimal Range')

            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel(self._get_sensor_unit(sensor), fontsize=10)
            ax.set_title(f"{sensor.replace('_', ' ').title()} Over Time",
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['figure_dpi'], bbox_inches='tight')
            print(f"✓ Sensor timeline saved to {save_path}")

        plt.show()

    def plot_sensor_correlation(self, save_path=None):
        """
        Plot correlation heatmap between sensors

        Args:
            save_path: Path to save figure
        """
        if self.sensor_data is None:
            print("No sensor data available")
            return

        # Select numerical columns
        sensor_cols = ['soil_moisture', 'temperature', 'humidity', 'light', 'ec', 'rainfall']
        df_corr = self.sensor_data[sensor_cols].corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   square=True, linewidths=1, cbar_kws={'label': 'Correlation'},
                   ax=ax)

        ax.set_title('Sensor Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['figure_dpi'], bbox_inches='tight')
            print(f"✓ Correlation heatmap saved to {save_path}")

        plt.show()

    def plot_crop_health_timeline(self, save_path=None):
        """
        Plot crop health predictions over time

        Args:
            save_path: Path to save figure
        """
        if self.predictions is None or 'health_class' not in self.predictions.columns:
            print("No crop health predictions available")
            return

        df = self.predictions.copy()

        # Map health classes to labels
        health_labels = {0: 'Critical', 1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
        df['health_label'] = df['health_class'].map(health_labels)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        # Color map
        colors = [self.viz_config['health_colors'][health_labels[i]] for i in df['health_class']]

        ax.scatter(df['timestamp'], df['health_class'], c=colors, s=20, alpha=0.6)
        ax.plot(df['timestamp'], df['health_class'].rolling(window=24, min_periods=1).mean(),
               color='black', linewidth=2, label='24h Moving Average')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Crop Health', fontsize=12)
        ax.set_title('Crop Health Over Time', fontsize=14, fontweight='bold')
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['Critical', 'Poor', 'Fair', 'Good', 'Excellent'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['figure_dpi'], bbox_inches='tight')
            print(f"✓ Crop health timeline saved to {save_path}")

        plt.show()

    def plot_alert_summary(self, save_path=None):
        """
        Plot alert statistics

        Args:
            save_path: Path to save figure
        """
        if self.alerts is None or len(self.alerts) == 0:
            print("No alerts available")
            return

        # Convert to DataFrame if list
        if isinstance(self.alerts, list):
            df = pd.DataFrame([a.to_dict() for a in self.alerts])
        else:
            df = self.alerts

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Alert counts by type
        alert_counts = df['alert_type'].value_counts()
        axes[0].barh(alert_counts.index, alert_counts.values, color='coral')
        axes[0].set_xlabel('Count', fontsize=12)
        axes[0].set_title('Alerts by Type', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Alerts over time
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        alerts_by_date = df.groupby('date').size()
        axes[1].plot(alerts_by_date.index, alerts_by_date.values, marker='o', linewidth=2)
        axes[1].fill_between(alerts_by_date.index, alerts_by_date.values, alpha=0.3)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Number of Alerts', fontsize=12)
        axes[1].set_title('Alerts Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['figure_dpi'], bbox_inches='tight')
            print(f"✓ Alert summary saved to {save_path}")

        plt.show()

    def plot_daily_summary(self, save_path=None):
        """
        Plot comprehensive daily summary dashboard

        Args:
            save_path: Path to save figure
        """
        if self.sensor_data is None:
            print("No sensor data available")
            return

        # Aggregate to daily
        df = self.sensor_data.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby('date').agg({
            'soil_moisture': 'mean',
            'temperature': ['mean', 'max', 'min'],
            'humidity': 'mean',
            'ec': 'mean',
            'rainfall': 'sum',
            'irrigation': 'sum'
        })

        # Create dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Soil moisture
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(daily.index, daily[('soil_moisture', 'mean')], linewidth=2, label='Soil Moisture')
        ax1.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Lower Threshold')
        ax1.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Upper Threshold')
        ax1.set_ylabel('Soil Moisture (%)')
        ax1.set_title('Daily Soil Moisture', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Temperature range
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.fill_between(daily.index,
                        daily[('temperature', 'min')],
                        daily[('temperature', 'max')],
                        alpha=0.3, label='Range')
        ax2.plot(daily.index, daily[('temperature', 'mean')], linewidth=2, label='Mean')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Daily Temperature', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Humidity
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(daily.index, daily[('humidity', 'mean')], linewidth=2, color='teal')
        ax3.set_ylabel('Humidity (%)')
        ax3.set_title('Daily Humidity', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. EC (nutrients)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(daily.index, daily[('ec', 'mean')], linewidth=2, color='purple')
        ax4.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
        ax4.set_ylabel('EC (dS/m)')
        ax4.set_title('Daily Nutrient Level (EC)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Rainfall + Irrigation
        ax5 = fig.add_subplot(gs[2, :])
        ax5.bar(daily.index, daily[('rainfall', 'sum')], label='Rainfall', alpha=0.7, color='blue')
        ax5.bar(daily.index, daily[('irrigation', 'sum')], label='Irrigation', alpha=0.7, color='cyan')
        ax5.set_ylabel('Water (mm)')
        ax5.set_xlabel('Date')
        ax5.set_title('Daily Rainfall and Irrigation', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle('Agricultural Dashboard - Daily Summary', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['figure_dpi'], bbox_inches='tight')
            print(f"✓ Daily summary dashboard saved to {save_path}")

        plt.show()

    def _get_sensor_unit(self, sensor):
        """Get unit label for sensor"""
        units = {
            'soil_moisture': 'Moisture (%)',
            'temperature': 'Temperature (°C)',
            'humidity': 'Humidity (%)',
            'light': 'PAR (µmol/m²/s)',
            'ec': 'EC (dS/m)',
            'rainfall': 'Rainfall (mm)',
            'irrigation': 'Irrigation (mm)'
        }
        return units.get(sensor, sensor)


def create_comprehensive_report(sensor_data, predictions, alerts, save_dir=None):
    """
    Create a comprehensive visualization report

    Args:
        sensor_data: DataFrame with sensor readings
        predictions: DataFrame with predictions
        alerts: List of alerts
        save_dir: Directory to save plots
    """
    save_dir = save_dir or config.VISUALIZATION_DIR
    os.makedirs(save_dir, exist_ok=True)

    dashboard = AgricultureDashboard(sensor_data, predictions, alerts)

    print("\nGenerating visualization report...")

    # 1. Sensor timelines
    dashboard.plot_sensor_timeline(save_path=os.path.join(save_dir, 'sensor_timeline.png'))

    # 2. Correlation heatmap
    dashboard.plot_sensor_correlation(save_path=os.path.join(save_dir, 'sensor_correlation.png'))

    # 3. Crop health timeline (if available)
    if predictions is not None and 'health_class' in predictions.columns:
        dashboard.plot_crop_health_timeline(save_path=os.path.join(save_dir, 'crop_health_timeline.png'))

    # 4. Alert summary (if available)
    if alerts is not None and len(alerts) > 0:
        dashboard.plot_alert_summary(save_path=os.path.join(save_dir, 'alert_summary.png'))

    # 5. Daily summary dashboard
    dashboard.plot_daily_summary(save_path=os.path.join(save_dir, 'daily_summary.png'))

    print(f"\n✓ All visualizations saved to {save_dir}")


if __name__ == "__main__":
    # Test visualization
    print("=" * 60)
    print("Visualization Dashboard - Test")
    print("=" * 60)

    # Generate sample data
    from data_generator import AgricultureDataGenerator

    generator = AgricultureDataGenerator(scenario='random', random_state=42)
    df = generator.generate(days=30, save_to_file=False)

    # Create dashboard
    dashboard = AgricultureDashboard(sensor_data=df)

    # Test plots
    print("\nGenerating test plots...")
    dashboard.plot_sensor_timeline(sensors=['soil_moisture', 'temperature'], days=30)
    dashboard.plot_sensor_correlation()
    dashboard.plot_daily_summary()

    print("\n✓ Visualization test complete")

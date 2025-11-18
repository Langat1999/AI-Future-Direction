"""
Alert System for Smart Agriculture
Monitors sensor data and generates actionable alerts
"""

import pandas as pd
from datetime import datetime, timedelta
import config


class Alert:
    """Represents a single alert"""

    def __init__(self, alert_type, severity, message, sensor_values, recommended_action, timestamp=None):
        """
        Create an alert

        Args:
            alert_type: Type of alert
            severity: Priority level ('low', 'medium', 'high', 'critical')
            message: Alert message
            sensor_values: Dictionary of relevant sensor readings
            recommended_action: Recommended action to take
            timestamp: Alert timestamp
        """
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.sensor_values = sensor_values
        self.recommended_action = recommended_action
        self.timestamp = timestamp or datetime.now()

    def to_dict(self):
        """Convert alert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'sensor_values': self.sensor_values,
            'recommended_action': self.recommended_action
        }

    def __str__(self):
        """String representation"""
        return f"[{self.severity.upper()}] {self.alert_type}: {self.message}"


class AlertSystem:
    """
    Monitors sensor data and generates alerts based on thresholds and rules
    """

    def __init__(self):
        """Initialize alert system"""
        self.alert_config = config.ALERT_CONFIG
        self.alert_history = []
        self.active_alerts = {}

    def check_conditions(self, sensor_data, timestamp=None):
        """
        Check sensor data against alert conditions

        Args:
            sensor_data: Dictionary or DataFrame row with sensor values
            timestamp: Current timestamp

        Returns:
            list: List of Alert objects
        """
        timestamp = timestamp or datetime.now()
        new_alerts = []

        # Convert DataFrame row to dict if needed
        if isinstance(sensor_data, pd.Series):
            sensor_data = sensor_data.to_dict()

        # Check each alert type
        for alert_type, alert_cfg in self.alert_config['alert_types'].items():
            if not alert_cfg['enabled']:
                continue

            alert = None

            if alert_type == 'drought_stress':
                alert = self._check_drought_stress(sensor_data, timestamp)
            elif alert_type == 'nutrient_deficiency':
                alert = self._check_nutrient_deficiency(sensor_data, timestamp)
            elif alert_type == 'heat_stress':
                alert = self._check_heat_stress(sensor_data, timestamp)
            elif alert_type == 'frost_warning':
                alert = self._check_frost_warning(sensor_data, timestamp)
            elif alert_type == 'disease_risk':
                alert = self._check_disease_risk(sensor_data, timestamp)
            elif alert_type == 'overwatering':
                alert = self._check_overwatering(sensor_data, timestamp)

            if alert:
                new_alerts.append(alert)
                self.alert_history.append(alert)
                self.active_alerts[alert_type] = alert

        return new_alerts

    def _check_drought_stress(self, data, timestamp):
        """Check for drought stress"""
        soil_moisture = data.get('soil_moisture', 50)

        if soil_moisture < 30:
            severity = 'high' if soil_moisture < 25 else 'medium'
            message = f"Soil moisture critically low at {soil_moisture:.1f}%"
            action = self.alert_config['alert_types']['drought_stress']['action']

            return Alert(
                alert_type='drought_stress',
                severity=severity,
                message=message,
                sensor_values={'soil_moisture': soil_moisture},
                recommended_action=action,
                timestamp=timestamp
            )
        return None

    def _check_nutrient_deficiency(self, data, timestamp):
        """Check for nutrient deficiency"""
        ec = data.get('ec', 2.0)

        if ec < 1.0:
            severity = 'high' if ec < 0.5 else 'medium'
            message = f"Electrical conductivity low at {ec:.2f} dS/m (nutrient deficiency)"
            action = self.alert_config['alert_types']['nutrient_deficiency']['action']

            return Alert(
                alert_type='nutrient_deficiency',
                severity=severity,
                message=message,
                sensor_values={'ec': ec},
                recommended_action=action,
                timestamp=timestamp
            )
        return None

    def _check_heat_stress(self, data, timestamp):
        """Check for heat stress"""
        temperature = data.get('temperature', 25)

        if temperature > 35:
            severity = 'high' if temperature > 38 else 'medium'
            message = f"Temperature high at {temperature:.1f}°C (heat stress risk)"
            action = self.alert_config['alert_types']['heat_stress']['action']

            return Alert(
                alert_type='heat_stress',
                severity=severity,
                message=message,
                sensor_values={'temperature': temperature},
                recommended_action=action,
                timestamp=timestamp
            )
        return None

    def _check_frost_warning(self, data, timestamp):
        """Check for frost risk"""
        temperature = data.get('temperature', 25)

        if temperature < 5:
            severity = 'high' if temperature < 2 else 'medium'
            message = f"Temperature low at {temperature:.1f}°C (frost risk)"
            action = self.alert_config['alert_types']['frost_warning']['action']

            return Alert(
                alert_type='frost_warning',
                severity=severity,
                message=message,
                sensor_values={'temperature': temperature},
                recommended_action=action,
                timestamp=timestamp
            )
        return None

    def _check_disease_risk(self, data, timestamp):
        """Check for disease risk (high humidity + moderate temp)"""
        humidity = data.get('humidity', 60)
        temperature = data.get('temperature', 25)

        if humidity > 80 and 20 <= temperature <= 25:
            message = f"High disease risk (humidity: {humidity:.1f}%, temp: {temperature:.1f}°C)"
            action = self.alert_config['alert_types']['disease_risk']['action']

            return Alert(
                alert_type='disease_risk',
                severity='medium',
                message=message,
                sensor_values={'humidity': humidity, 'temperature': temperature},
                recommended_action=action,
                timestamp=timestamp
            )
        return None

    def _check_overwatering(self, data, timestamp):
        """Check for overwatering"""
        soil_moisture = data.get('soil_moisture', 50)

        if soil_moisture > 75:
            severity = 'medium' if soil_moisture < 85 else 'high'
            message = f"Soil moisture high at {soil_moisture:.1f}% (overwatering risk)"
            action = self.alert_config['alert_types']['overwatering']['action']

            return Alert(
                alert_type='overwatering',
                severity=severity,
                message=message,
                sensor_values={'soil_moisture': soil_moisture},
                recommended_action=action,
                timestamp=timestamp
            )
        return None

    def get_alert_summary(self):
        """Get summary of all alerts"""
        if not self.alert_history:
            return {"total": 0, "by_type": {}, "by_severity": {}}

        df = pd.DataFrame([a.to_dict() for a in self.alert_history])

        summary = {
            'total': len(self.alert_history),
            'by_type': df['alert_type'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'first_alert': df['timestamp'].min(),
            'last_alert': df['timestamp'].max()
        }

        return summary

    def get_recent_alerts(self, hours=24):
        """Get alerts from last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alert_history if a.timestamp >= cutoff]
        return recent

    def clear_alert(self, alert_type):
        """Clear an active alert"""
        if alert_type in self.active_alerts:
            del self.active_alerts[alert_type]

    def save_alerts(self, filepath=None):
        """Save alert history to CSV"""
        filepath = filepath or config.OUTPUT_CONFIG['alerts_file']

        if not self.alert_history:
            print("No alerts to save")
            return

        df = pd.DataFrame([a.to_dict() for a in self.alert_history])
        df.to_csv(filepath, index=False)
        print(f"✓ Alerts saved to {filepath}")

    def load_alerts(self, filepath=None):
        """Load alert history from CSV"""
        filepath = filepath or config.OUTPUT_CONFIG['alerts_file']

        try:
            df = pd.DataFrame(filepath)
            self.alert_history = []

            for _, row in df.iterrows():
                alert = Alert(
                    alert_type=row['alert_type'],
                    severity=row['severity'],
                    message=row['message'],
                    sensor_values=eval(row['sensor_values']),  # Convert string to dict
                    recommended_action=row['recommended_action'],
                    timestamp=pd.to_datetime(row['timestamp'])
                )
                self.alert_history.append(alert)

            print(f"✓ Loaded {len(self.alert_history)} alerts from {filepath}")

        except Exception as e:
            print(f"Error loading alerts: {e}")


class IrrigationDecisionSystem:
    """
    Rule-based irrigation decision system
    """

    def __init__(self):
        """Initialize irrigation decision system"""
        pass

    def recommend(self, sensor_data):
        """
        Recommend irrigation action

        Args:
            sensor_data: Dictionary with sensor readings

        Returns:
            dict: Irrigation recommendation
        """
        soil_moisture = sensor_data.get('soil_moisture', 50)
        temperature = sensor_data.get('temperature', 25)
        rainfall_recent = sensor_data.get('rainfall_7d', 0)

        # Decision logic
        if soil_moisture < 25:
            action = "IRRIGATE URGENTLY"
            amount = 30  # mm
            priority = "critical"
            reason = f"Severe drought stress (moisture: {soil_moisture:.1f}%)"

        elif soil_moisture < 35:
            action = "IRRIGATE SOON"
            amount = 20  # mm
            priority = "high"
            reason = f"Low soil moisture ({soil_moisture:.1f}%)"

        elif soil_moisture < 40 and temperature > 30:
            action = "IRRIGATE MODERATELY"
            amount = 15  # mm
            priority = "medium"
            reason = f"Moderate moisture + high temp"

        elif soil_moisture > 70:
            action = "NO IRRIGATION - OVERWATERED"
            amount = 0
            priority = "low"
            reason = f"Soil moisture high ({soil_moisture:.1f}%)"

        elif rainfall_recent > 40:
            action = "NO IRRIGATION - RECENT RAIN"
            amount = 0
            priority = "low"
            reason = f"Recent rainfall adequate ({rainfall_recent:.0f}mm)"

        else:
            action = "NO ACTION NEEDED"
            amount = 0
            priority = "low"
            reason = f"Soil moisture optimal ({soil_moisture:.1f}%)"

        return {
            'action': action,
            'amount_mm': amount,
            'priority': priority,
            'reason': reason,
            'soil_moisture': soil_moisture
        }


class FertilizerDecisionSystem:
    """
    Rule-based fertilizer decision system
    """

    def __init__(self):
        """Initialize fertilizer decision system"""
        pass

    def recommend(self, sensor_data, days_since_planting):
        """
        Recommend fertilizer application

        Args:
            sensor_data: Dictionary with sensor readings
            days_since_planting: Days since crop was planted

        Returns:
            dict: Fertilizer recommendation
        """
        ec = sensor_data.get('ec', 2.0)

        # Decision logic
        if days_since_planting < 0:
            action = "APPLY AT PLANTING"
            amount = "100 kg/ha NPK (10-10-10)"
            priority = "high"
            reason = "Pre-planting fertilization"

        elif ec < 0.8:
            action = "APPLY IMMEDIATELY"
            amount = "80 kg/ha NPK (15-15-15)"
            priority = "critical"
            reason = f"Severe nutrient deficiency (EC: {ec:.2f})"

        elif ec < 1.2:
            action = "APPLY SOON"
            amount = "60 kg/ha NPK (15-15-15)"
            priority = "high"
            reason = f"Nutrient deficiency (EC: {ec:.2f})"

        elif days_since_planting % 45 == 0 and ec < 2.5:
            action = "SCHEDULED APPLICATION"
            amount = "40 kg/ha NPK (20-20-20)"
            priority = "medium"
            reason = "Scheduled side-dressing"

        else:
            action = "NO APPLICATION NEEDED"
            amount = "0 kg/ha"
            priority = "low"
            reason = f"Nutrient levels adequate (EC: {ec:.2f})"

        return {
            'action': action,
            'amount': amount,
            'priority': priority,
            'reason': reason,
            'ec': ec
        }


if __name__ == "__main__":
    # Test alert system
    print("=" * 60)
    print("Alert System - Test")
    print("=" * 60)

    alert_system = AlertSystem()

    # Test scenarios
    test_scenarios = [
        {'name': 'Normal', 'soil_moisture': 50, 'temperature': 25, 'humidity': 60, 'ec': 2.5},
        {'name': 'Drought', 'soil_moisture': 25, 'temperature': 35, 'humidity': 40, 'ec': 2.0},
        {'name': 'Nutrient Def', 'soil_moisture': 50, 'temperature': 25, 'humidity': 60, 'ec': 0.8},
        {'name': 'Disease Risk', 'soil_moisture': 50, 'temperature': 22, 'humidity': 85, 'ec': 2.0},
        {'name': 'Overwatering', 'soil_moisture': 80, 'temperature': 20, 'humidity': 75, 'ec': 1.5},
    ]

    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Sensors: Moisture={scenario['soil_moisture']}%, "
              f"Temp={scenario['temperature']}°C, "
              f"Humidity={scenario['humidity']}%, "
              f"EC={scenario['ec']}")

        alerts = alert_system.check_conditions(scenario)

        if alerts:
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("  No alerts")

    # Summary
    print("\n" + "=" * 60)
    print("Alert Summary")
    print("=" * 60)
    summary = alert_system.get_alert_summary()
    print(f"Total alerts: {summary['total']}")
    print(f"By type: {summary['by_type']}")
    print(f"By severity: {summary['by_severity']}")

    # Test irrigation recommendations
    print("\n" + "=" * 60)
    print("Irrigation Recommendations")
    print("=" * 60)

    irrigation_system = IrrigationDecisionSystem()

    for scenario in test_scenarios[:3]:
        rec = irrigation_system.recommend(scenario)
        print(f"\n{scenario['name']}:")
        print(f"  Action: {rec['action']}")
        print(f"  Amount: {rec['amount_mm']} mm")
        print(f"  Reason: {rec['reason']}")

    print("\n✓ Alert system test complete")

"""
AI Prediction Models for Smart Agriculture
Crop health classification and yield prediction
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
import config


class CropHealthPredictor:
    """
    Predicts crop health status based on sensor readings
    Multi-class classification: Critical, Poor, Fair, Good, Excellent
    """

    def __init__(self, random_state=None):
        """Initialize crop health predictor"""
        self.random_state = random_state or config.RANDOM_SEED
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = config.CROP_HEALTH_CONFIG['features']
        self.classes = config.CROP_HEALTH_CONFIG['classes']

    def _label_health(self, df):
        """
        Generate health labels based on sensor conditions

        Args:
            df: DataFrame with sensor data

        Returns:
            np.array: Health labels (0-4)
        """
        labels = np.zeros(len(df))

        for idx, row in df.iterrows():
            score = 0

            # Soil moisture score (0-2)
            if 40 <= row['soil_moisture'] <= 60:
                score += 2
            elif 30 <= row['soil_moisture'] < 40 or 60 < row['soil_moisture'] <= 70:
                score += 1

            # Temperature score (0-1)
            if 20 <= row['temperature'] <= 30:
                score += 1
            elif row['temperature'] > 35 or row['temperature'] < 10:
                score -= 1

            # EC score (0-1)
            if 1.5 <= row['ec'] <= 3.0:
                score += 1
            elif row['ec'] < 1.0:
                score -= 1

            # Map score to health class
            if score >= 4:
                labels[idx] = 4  # Excellent
            elif score >= 3:
                labels[idx] = 3  # Good
            elif score >= 2:
                labels[idx] = 2  # Fair
            elif score >= 1:
                labels[idx] = 1  # Poor
            else:
                labels[idx] = 0  # Critical

        return labels.astype(int)

    def train(self, df, save_model=True):
        """
        Train crop health classification model

        Args:
            df: DataFrame with sensor data
            save_model: Whether to save trained model

        Returns:
            dict: Training metrics
        """
        print("\nTraining Crop Health Classifier...")

        # Generate labels
        y = self._label_health(df)

        # Extract features
        X = df[self.feature_names].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=config.CROP_HEALTH_CONFIG['n_estimators'],
            max_depth=config.CROP_HEALTH_CONFIG['max_depth'],
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✓ Training complete - Accuracy: {accuracy:.4f}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=[self.classes[i] for i in range(5)]))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance.to_string(index=False))

        # Save model
        if save_model:
            self._save_model()

        metrics = {
            'accuracy': accuracy,
            'feature_importance': feature_importance.to_dict('records')
        }

        return metrics

    def predict(self, features):
        """
        Predict crop health for given features

        Args:
            features: Feature vector or DataFrame

        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to numpy array if needed
        if isinstance(features, dict):
            features = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            features = features[self.feature_names].values

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        result = {
            'health_class': int(prediction),
            'health_label': self.classes[prediction],
            'confidence': float(probabilities[prediction]),
            'probabilities': {self.classes[i]: float(probabilities[i]) for i in range(5)}
        }

        return result

    def predict_batch(self, df):
        """
        Predict crop health for multiple samples

        Args:
            df: DataFrame with features

        Returns:
            np.array: Predictions
        """
        features = df[self.feature_names].values
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return predictions

    def _save_model(self):
        """Save model and scaler to disk"""
        model_path = os.path.join(config.MODEL_DIR, 'crop_health_model.pkl')
        scaler_path = os.path.join(config.MODEL_DIR, 'health_scaler.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"✓ Model saved to {model_path}")

    def load_model(self):
        """Load trained model from disk"""
        model_path = os.path.join(config.MODEL_DIR, 'crop_health_model.pkl')
        scaler_path = os.path.join(config.MODEL_DIR, 'health_scaler.pkl')

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"✓ Model loaded from {model_path}")


class YieldPredictor:
    """
    Predicts crop yield (kg/ha) based on seasonal sensor aggregates
    """

    def __init__(self, random_state=None):
        """Initialize yield predictor"""
        self.random_state = random_state or config.RANDOM_SEED
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = config.YIELD_PREDICTOR_CONFIG['features']

    def _generate_yield_labels(self, df):
        """
        Generate synthetic yield labels based on growing conditions

        Args:
            df: DataFrame with aggregated features

        Returns:
            np.array: Yield values (kg/ha)
        """
        # Base yield
        base_yield = 8000  # kg/ha for corn

        yields = []

        for idx, row in df.iterrows():
            yield_val = base_yield

            # Soil moisture effect
            moisture_optimal = (row['avg_soil_moisture'] - 50) ** 2
            yield_val -= moisture_optimal * 2

            # Temperature effect
            temp_optimal = max(0, abs(row['avg_temperature'] - 25) - 5)
            yield_val -= temp_optimal * 50

            # Nutrient effect
            if row['avg_ec'] < 1.5:
                yield_val -= (1.5 - row['avg_ec']) * 1000
            elif row['avg_ec'] > 3.5:
                yield_val -= (row['avg_ec'] - 3.5) * 800

            # Rainfall effect
            if row['total_rainfall'] < 400:
                yield_val -= (400 - row['total_rainfall']) * 5
            elif row['total_rainfall'] > 800:
                yield_val -= (row['total_rainfall'] - 800) * 3

            # Stress penalty
            yield_val -= row['stress_days'] * 50

            # Add some noise
            noise = np.random.normal(0, 300)
            yield_val += noise

            # Clip to reasonable range
            yield_val = np.clip(yield_val, 3000, 12000)

            yields.append(yield_val)

        return np.array(yields)

    def _aggregate_features(self, df):
        """
        Aggregate sensor data to seasonal features

        Args:
            df: DataFrame with time-series sensor data

        Returns:
            pd.DataFrame: Aggregated features
        """
        # Group by season if available, otherwise treat as single season
        if 'season' in df.columns:
            grouped = df.groupby('season')
        else:
            df['season'] = 1
            grouped = df.groupby('season')

        aggregated = grouped.agg({
            'soil_moisture': 'mean',
            'temperature': 'mean',
            'ec': 'mean',
            'rainfall': 'sum',
            'drought_stress': 'sum',
            'heat_stress': 'sum',
            'nutrient_stress': 'sum',
            'days_since_planting': 'max'
        }).reset_index()

        # Rename columns
        aggregated.columns = [
            'season',
            'avg_soil_moisture',
            'avg_temperature',
            'avg_ec',
            'total_rainfall',
            'drought_stress_days',
            'heat_stress_days',
            'nutrient_stress_days',
            'growth_period_days'
        ]

        # Calculate total stress days
        aggregated['stress_days'] = (aggregated['drought_stress_days'] +
                                    aggregated['heat_stress_days'] +
                                    aggregated['nutrient_stress_days'])

        # Calculate optimal days (inverse of stress)
        max_days = aggregated['growth_period_days'].max()
        aggregated['optimal_days'] = max_days - aggregated['stress_days']

        return aggregated

    def train(self, df, save_model=True):
        """
        Train yield prediction model

        Args:
            df: DataFrame with sensor data
            save_model: Whether to save trained model

        Returns:
            dict: Training metrics
        """
        print("\nTraining Yield Predictor...")

        # Aggregate features
        df_agg = self._aggregate_features(df)

        # Generate labels
        y = self._generate_yield_labels(df_agg)

        # Extract features
        X = df_agg[self.feature_names].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Gradient Boosting
        self.model = GradientBoostingRegressor(
            n_estimators=config.YIELD_PREDICTOR_CONFIG['n_estimators'],
            learning_rate=config.YIELD_PREDICTOR_CONFIG['learning_rate'],
            max_depth=config.YIELD_PREDICTOR_CONFIG['max_depth'],
            random_state=self.random_state
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"✓ Training complete - R² Score: {r2:.4f}, RMSE: {rmse:.0f} kg/ha")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance.to_string(index=False))

        # Save model
        if save_model:
            self._save_model()

        metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'feature_importance': feature_importance.to_dict('records')
        }

        return metrics

    def predict(self, features):
        """
        Predict yield for given features

        Args:
            features: Feature vector, dict, or DataFrame

        Returns:
            float: Predicted yield (kg/ha)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to numpy array if needed
        if isinstance(features, dict):
            features = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            features = features[self.feature_names].values

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]

        return float(prediction)

    def _save_model(self):
        """Save model and scaler to disk"""
        model_path = os.path.join(config.MODEL_DIR, 'yield_model.pkl')
        scaler_path = os.path.join(config.MODEL_DIR, 'yield_scaler.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"✓ Model saved to {model_path}")

    def load_model(self):
        """Load trained model from disk"""
        model_path = os.path.join(config.MODEL_DIR, 'yield_model.pkl')
        scaler_path = os.path.join(config.MODEL_DIR, 'yield_scaler.pkl')

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"✓ Model loaded from {model_path}")


if __name__ == "__main__":
    # Test AI models
    print("=" * 60)
    print("AI Predictor - Test")
    print("=" * 60)

    # Generate sample data
    from data_generator import AgricultureDataGenerator

    generator = AgricultureDataGenerator(scenario='random', random_state=42)
    df = generator.generate(days=365, save_to_file=False)

    # Test Crop Health Predictor
    print("\n" + "=" * 60)
    print("Testing Crop Health Predictor")
    print("=" * 60)

    health_predictor = CropHealthPredictor()
    health_metrics = health_predictor.train(df, save_model=False)

    # Test single prediction
    sample_features = {
        'soil_moisture': 50,
        'temperature': 25,
        'humidity': 60,
        'light': 800,
        'ec': 2.5,
        'days_since_planting': 60,
        'rainfall_7d': 40
    }

    result = health_predictor.predict(sample_features)
    print(f"\nSample Prediction:")
    print(f"  Health: {result['health_label']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # Test Yield Predictor
    print("\n" + "=" * 60)
    print("Testing Yield Predictor")
    print("=" * 60)

    # Generate multi-season data for yield prediction
    df['season'] = 1
    seasons = [df.copy() for _ in range(5)]
    for i, season_df in enumerate(seasons):
        season_df['season'] = i + 1
    multi_season_df = pd.concat(seasons, ignore_index=True)

    yield_predictor = YieldPredictor()
    yield_metrics = yield_predictor.train(multi_season_df, save_model=False)

    print("\n✓ AI predictor test complete")

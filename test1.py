import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class TrafficManagementSystem:
    def __init__(self):
        self.le_intersection = LabelEncoder()
        self.le_day = LabelEncoder()
        self.le_weather = LabelEncoder()
        self.scaler = StandardScaler()
        self.traffic_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.congestion_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def generate_sample_data(self, n_samples=5000):
        """Generate synthetic traffic data with realistic patterns"""
        
        # Create intersection grid (A1 to Z24)
        intersections = []
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for number in range(1, 25):
                intersections.append(f"{letter}{number}")
        
        data = []
        
        for _ in range(n_samples):
            intersection = random.choice(intersections)
            
            # Extract coordinates for distance calculations
            row = ord(intersection[0]) - ord('A')
            col = int(intersection[1:]) - 1
            
            # Days of week
            day = random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                               'Friday', 'Saturday', 'Sunday'])
            
            # Hour of day (0-23)
            hour = random.randint(0, 23)
            
            # Weather conditions
            weather = random.choice(['Clear', 'Rain', 'Snow', 'Fog'])
            
            # Traffic patterns based on realistic scenarios
            is_weekend = day in ['Saturday', 'Sunday']
            is_rush_hour = hour in [7, 8, 9, 17, 18, 19]  # Morning and evening rush
            is_business_district = row < 10 and col < 12  # A1-J12 area
            is_residential = row >= 15  # P1-Z24 area
            
            # Base congestion time with realistic patterns
            base_congestion = 2.0
            
            # Rush hour increases congestion
            if is_rush_hour:
                base_congestion *= random.uniform(2.5, 4.0)
            
            # Weekend patterns
            if is_weekend:
                if hour in [10, 11, 12, 13, 14]:  # Weekend shopping hours
                    base_congestion *= random.uniform(1.5, 2.5)
                else:
                    base_congestion *= random.uniform(0.3, 0.8)
            
            # Location-based patterns
            if is_business_district and not is_weekend:
                base_congestion *= random.uniform(1.3, 2.0)
            elif is_residential:
                base_congestion *= random.uniform(0.7, 1.2)
            
            # Weather impact
            weather_multiplier = {
                'Clear': 1.0,
                'Rain': random.uniform(1.3, 1.8),
                'Snow': random.uniform(1.5, 2.2),
                'Fog': random.uniform(1.2, 1.6)
            }
            base_congestion *= weather_multiplier[weather]
            
            # Add some randomness
            congestion_time = max(0.5, base_congestion + np.random.normal(0, 1))
            
            # Vehicle count (correlated with congestion)
            vehicle_count = max(1, int(congestion_time * random.uniform(3, 8)))
            
            # Average speed (inversely correlated with congestion)
            avg_speed = max(5, 45 - (congestion_time * random.uniform(2, 5)))
            
            # Distance from city center (A1)
            distance_from_center = np.sqrt((row**2) + (col**2))
            
            # Emergency vehicle presence (rare event)
            emergency_vehicle = random.random() < 0.05
            
            # Special event nearby (rare)
            special_event = random.random() < 0.03
            
            # Traffic light action (target variable)
            # Priority scoring system
            priority_score = congestion_time
            
            if emergency_vehicle:
                priority_score += 10  # High priority for emergency vehicles
            if special_event:
                priority_score += 5
            if is_rush_hour:
                priority_score += 2
            if weather != 'Clear':
                priority_score += 1
                
            # Determine action based on priority score
            if priority_score > 8:
                action = 'Open_Immediately'
            elif priority_score > 5:
                action = 'Open_Soon'
            elif priority_score > 3:
                action = 'Normal_Schedule'
            else:
                action = 'Delay_Opening'
            
            data.append({
                'intersection': intersection,
                'day': day,
                'hour': hour,
                'congestion_time_minutes': round(congestion_time, 2),
                'vehicle_count': vehicle_count,
                'average_speed_kmh': round(avg_speed, 1),
                'weather': weather,
                'distance_from_center': round(distance_from_center, 2),
                'emergency_vehicle_present': emergency_vehicle,
                'special_event_nearby': special_event,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,
                'traffic_light_action': action
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        df_encoded = df.copy()
        
        # Encode categorical variables
        df_encoded['intersection_encoded'] = self.le_intersection.fit_transform(df['intersection'])
        df_encoded['day_encoded'] = self.le_day.fit_transform(df['day'])
        df_encoded['weather_encoded'] = self.le_weather.fit_transform(df['weather'])
        
        # Select features for modeling
        feature_columns = [
            'intersection_encoded', 'day_encoded', 'hour', 'congestion_time_minutes',
            'vehicle_count', 'average_speed_kmh', 'weather_encoded', 
            'distance_from_center', 'emergency_vehicle_present', 'special_event_nearby',
            'is_weekend', 'is_rush_hour'
        ]
        
        X = df_encoded[feature_columns]
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_columns
    
    def train_models(self, df):
        """Train both classification and regression models"""
        print("Preparing features...")
        X, feature_columns = self.prepare_features(df)
        
        # Classification model for traffic light action
        y_classification = df['traffic_light_action']
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
        )
        
        print("Training traffic light action classifier...")
        self.traffic_classifier.fit(X_train_clf, y_train_clf)
        
        # Regression model for congestion prediction
        y_regression = df['congestion_time_minutes']
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )
        
        print("Training congestion time predictor...")
        self.congestion_predictor.fit(X_train_reg, y_train_reg)
        
        # Evaluate models
        print("\n=== TRAFFIC LIGHT ACTION CLASSIFIER RESULTS ===")
        y_pred_clf = self.traffic_classifier.predict(X_test_clf)
        print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test_clf, y_pred_clf))
        
        print("\n=== CONGESTION TIME PREDICTOR RESULTS ===")
        y_pred_reg = self.congestion_predictor.predict(X_test_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse:.3f} minutes")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_test_reg - y_pred_reg)):.3f} minutes")
        
        # Feature importance
        print("\n=== FEATURE IMPORTANCE (Classification) ===")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.traffic_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance)
        
        return X_test_clf, y_test_clf, y_pred_clf, X_test_reg, y_test_reg, y_pred_reg
    
    def predict_traffic_action(self, intersection, day, hour, congestion_time, 
                             vehicle_count, avg_speed, weather, emergency=False, 
                             special_event=False):
        """Predict traffic light action for a specific scenario"""
        
        # Calculate derived features
        row = ord(intersection[0]) - ord('A')
        col = int(intersection[1:]) - 1
        distance_from_center = np.sqrt((row**2) + (col**2))
        is_weekend = day in ['Saturday', 'Sunday']
        is_rush_hour = hour in [7, 8, 9, 17, 18, 19]
        
        # Prepare input data
        input_data = pd.DataFrame({
            'intersection': [intersection],
            'day': [day],
            'hour': [hour],
            'congestion_time_minutes': [congestion_time],
            'vehicle_count': [vehicle_count],
            'average_speed_kmh': [avg_speed],
            'weather': [weather],
            'distance_from_center': [distance_from_center],
            'emergency_vehicle_present': [emergency],
            'special_event_nearby': [special_event],
            'is_weekend': [is_weekend],
            'is_rush_hour': [is_rush_hour]
        })
        
        # Encode features
        input_encoded = input_data.copy()
        input_encoded['intersection_encoded'] = self.le_intersection.transform([intersection])
        input_encoded['day_encoded'] = self.le_day.transform([day])
        input_encoded['weather_encoded'] = self.le_weather.transform([weather])
        
        feature_columns = [
            'intersection_encoded', 'day_encoded', 'hour', 'congestion_time_minutes',
            'vehicle_count', 'average_speed_kmh', 'weather_encoded', 
            'distance_from_center', 'emergency_vehicle_present', 'special_event_nearby',
            'is_weekend', 'is_rush_hour'
        ]
        
        X_input = input_encoded[feature_columns]
        X_input_scaled = self.scaler.transform(X_input)
        
        # Make predictions
        action_prediction = self.traffic_classifier.predict(X_input_scaled)[0]
        action_probability = self.traffic_classifier.predict_proba(X_input_scaled)[0]
        congestion_prediction = self.congestion_predictor.predict(X_input_scaled)[0]
        
        return {
            'recommended_action': action_prediction,
            'action_probabilities': dict(zip(self.traffic_classifier.classes_, action_probability)),
            'predicted_congestion_time': round(congestion_prediction, 2)
        }

# Main execution
if __name__ == "__main__":
    print("🚦 Smart Traffic Management System")
    print("=" * 50)
    
    # Initialize system
    tms = TrafficManagementSystem()
    
    # Generate sample data
    print("Generating sample traffic data...")
    df = tms.generate_sample_data(n_samples=5000)
    
    # Display sample data
    print("\n📊 Sample Data Overview:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget distribution:")
    print(df['traffic_light_action'].value_counts())
    
    # Train models
    print("\n🤖 Training Models...")
    results = tms.train_models(df)
    
    # Example predictions
    print("\n🔮 Example Predictions:")
    print("-" * 30)
    
    # Scenario 1: Morning rush hour in business district
    result1 = tms.predict_traffic_action(
        intersection='E5', day='Monday', hour=8, congestion_time=6.5,
        vehicle_count=25, avg_speed=15, weather='Clear'
    )
    print("Scenario 1 - Morning Rush (E5, Monday 8AM):")
    print(f"  Action: {result1['recommended_action']}")
    print(f"  Predicted congestion: {result1['predicted_congestion_time']} minutes")
    
    # Scenario 2: Weekend with emergency vehicle
    result2 = tms.predict_traffic_action(
        intersection='M12', day='Saturday', hour=14, congestion_time=3.2,
        vehicle_count=12, avg_speed=35, weather='Rain', emergency=True
    )
    print("\nScenario 2 - Weekend Emergency (M12, Saturday 2PM, Rain):")
    print(f"  Action: {result2['recommended_action']}")
    print(f"  Predicted congestion: {result2['predicted_congestion_time']} minutes")
    
    # Scenario 3: Late night residential area
    result3 = tms.predict_traffic_action(
        intersection='T20', day='Wednesday', hour=23, congestion_time=1.5,
        vehicle_count=3, avg_speed=45, weather='Clear'
    )
    print("\nScenario 3 - Late Night Residential (T20, Wednesday 11PM):")
    print(f"  Action: {result3['recommended_action']}")
    print(f"  Predicted congestion: {result3['predicted_congestion_time']} minutes")
    
    print("\n✅ Traffic Management System Ready!")
    print("The system can now predict optimal traffic light actions based on real-time conditions.")
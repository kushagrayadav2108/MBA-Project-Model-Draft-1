"""
Seed Demand Prediction Model using Ridge Regression
Predicts seed demand ranges for each variety based on meteorological and market data
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import pickle
from pathlib import Path

from data_processor import prepare_data_for_modeling

warnings.filterwarnings('ignore')


class SeedDemandPredictor:
    """
    Linear Regression model to predict seed demand for each rice variety.
    """
    
    VARIETIES = ['Pb_1121', 'Pb_1718', 'Pb_1885', 'Pb_1509', 'Pb_1692', 'Pb_1847']
    
    # Variety duration information (for meteorological sensitivity)
    VARIETY_INFO = {
        'Pb_1121': {'duration': 'long', 'days': '135-145', 'category': 'Long Duration'},
        'Pb_1718': {'duration': 'long', 'days': '135-145', 'category': 'Long Duration'},
        'Pb_1885': {'duration': 'long', 'days': '135-145', 'category': 'Long Duration'},
        'Pb_1509': {'duration': 'short', 'days': '110-120', 'category': 'Short Duration'},
        'Pb_1692': {'duration': 'short', 'days': '110-120', 'category': 'Short Duration'},
        'Pb_1847': {'duration': 'short', 'days': '110-120', 'category': 'Short Duration'},
    }
    
    FEATURE_COLS = [
        'Max_Temp', 'Min_Temp', 'Pre_Monsoon_Rainfall', 
        'Monsoon_Rainfall', 'Post_Monsoon_Rainfall', 'Monsoon_Duration'
    ]
    
    # Seeding rate: 5-8 kg/Acre = ~12.5-20 kg/hectare, using midpoint ~15 kg/ha
    SEEDING_RATE_KG_PER_HA = 15.0  # kg per hectare (approximately 6 kg/acre)
    
    def __init__(self):
        self.models = {}  # One model per variety for share prediction
        self.scalers = {}  # Feature scalers
        self.mean_total_area = None  # Historical mean total area
        self.historical_shares = {}  # Historical share statistics
        self.is_trained = False
        
    def prepare_training_data(self, df: pd.DataFrame) -> dict:
        """
        Prepare training data for each variety.
        Returns dict with variety -> (X, y) pairs
        """
        training_data = {}
        
        for variety in self.VARIETIES:
            share_col = f'{variety}_Share'
            
            # Get rows where we have both features and target
            mask = df[share_col].notna()
            for col in self.FEATURE_COLS:
                mask = mask & df[col].notna()
            
            subset = df[mask].copy()
            
            if len(subset) >= 3:  # Need at least 3 data points
                X = subset[self.FEATURE_COLS].values
                y = subset[share_col].values
                training_data[variety] = (X, y, subset)
                
        return training_data
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train linear regression models for each variety.
        
        Args:
            df: Processed DataFrame from data_processor
            
        Returns:
            Dictionary with training metrics
        """
        training_data = self.prepare_training_data(df)
        metrics = {}
        
        # Calculate mean total area from historical data
        valid_area = df['Total_Area_Hectare'].dropna()
        if len(valid_area) > 0:
            self.mean_total_area = valid_area.mean()
        else:
            self.mean_total_area = 20000  # Default fallback
        
        for variety in self.VARIETIES:
            if variety in training_data:
                X, y, subset = training_data[variety]
                
                # Store historical share statistics with minimum std for reasonable ranges
                min_std = 3.0  # Minimum 3% standard deviation
                self.historical_shares[variety] = {
                    'mean': y.mean(),
                    'std': max(y.std(), min_std),
                    'min': y.min(),
                    'max': y.max()
                }
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[variety] = scaler
                
                # Train model with Ridge regression (prevents overfitting on small datasets)
                model = Ridge(alpha=10.0)
                model.fit(X_scaled, y)
                self.models[variety] = model
                
                # Calculate metrics
                y_pred = model.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                metrics[variety] = {
                    'r2': r2,
                    'mae': mae,
                    'n_samples': len(y),
                    'mean_share': y.mean(),
                    'residual_std': np.std(y - y_pred)
                }
            else:
                # For varieties with insufficient data, use historical statistics
                # Prefer recent years (last 3-5 years) for better relevance
                share_col = f'{variety}_Share'
                valid_shares = df[share_col].dropna()
                
                if len(valid_shares) > 0:
                    # Use most recent 3 years if available for better prediction
                    recent_df = df.nlargest(5, 'Year')
                    recent_shares = recent_df[share_col].dropna()
                    
                    if len(recent_shares) >= 2:
                        use_shares = recent_shares
                    else:
                        use_shares = valid_shares
                    
                    # Calculate statistics with minimum std to avoid zero-width confidence intervals
                    calculated_std = use_shares.std() if len(use_shares) > 1 else use_shares.mean() * 0.2
                    min_std = 3.0  # Minimum 3% standard deviation for reasonable ranges
                    
                    self.historical_shares[variety] = {
                        'mean': use_shares.mean(),
                        'std': max(calculated_std, min_std),
                        'min': valid_shares.min(),
                        'max': valid_shares.max()
                    }
                else:
                    # Default for new varieties
                    self.historical_shares[variety] = {
                        'mean': 10.0,
                        'std': 5.0,
                        'min': 5.0,
                        'max': 20.0
                    }
                metrics[variety] = {
                    'r2': None,
                    'mae': None,
                    'n_samples': 0,
                    'mean_share': self.historical_shares[variety]['mean'],
                    'note': 'Using historical mean (insufficient data for regression)'
                }
        
        self.is_trained = True
        return metrics
    
    def predict(self, input_data: dict, confidence_level: float = 0.95) -> dict:
        """
        Predict seed demand ranges for each variety.
        
        Args:
            input_data: Dictionary with meteorological features
                - Max_Temp, Min_Temp, Pre_Monsoon_Rainfall,
                  Monsoon_Rainfall, Post_Monsoon_Rainfall, Monsoon_Duration
            confidence_level: Confidence level for prediction interval (default 0.95)
            
        Returns:
            Dictionary with predictions for each variety
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare input features
        X = np.array([[
            input_data.get('Max_Temp', 43.0),
            input_data.get('Min_Temp', 21.0),
            input_data.get('Pre_Monsoon_Rainfall', 50.0),
            input_data.get('Monsoon_Rainfall', 200.0),
            input_data.get('Post_Monsoon_Rainfall', 30.0),
            input_data.get('Monsoon_Duration', 90.0)
        ]])
        
        # Use provided or estimated total area
        total_area = input_data.get('Total_Area', self.mean_total_area)
        
        # Z-score for confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 1.645
        
        # First pass: collect raw predictions
        raw_predictions = {}
        
        for variety in self.VARIETIES:
            hist_stats = self.historical_shares[variety]
            
            if variety in self.models:
                # Scale and predict
                X_scaled = self.scalers[variety].transform(X)
                raw_prediction = self.models[variety].predict(X_scaled)[0]
                
                # Sanity check: if prediction is wildly outside historical range, use historical mean
                historical_range = hist_stats['max'] - hist_stats['min']
                if raw_prediction < hist_stats['min'] - historical_range or raw_prediction > hist_stats['max'] + historical_range:
                    # Prediction is unreasonable, fall back to historical mean
                    predicted_share = hist_stats['mean']
                else:
                    predicted_share = raw_prediction
                
                # Bound the share to reasonable range
                predicted_share = np.clip(predicted_share, 0, 100)
            else:
                # Use historical statistics
                predicted_share = hist_stats['mean']
            
            raw_predictions[variety] = {
                'predicted_share': predicted_share,
                'std': hist_stats['std']
            }
        
        # Normalize shares to sum to ~95% (leaving 5% for Others)
        total_raw_share = sum(p['predicted_share'] for p in raw_predictions.values())
        target_total = 95.0  # Target sum (leaving 5% for Others)
        
        if total_raw_share > 0:
            normalization_factor = target_total / total_raw_share
        else:
            normalization_factor = 1.0
        
        # Second pass: create normalized predictions
        predictions = {}
        
        for variety in self.VARIETIES:
            raw = raw_predictions[variety]
            predicted_share = raw['predicted_share'] * normalization_factor
            share_std = raw['std'] * normalization_factor
            
            share_min = max(0, predicted_share - z_score * share_std)
            share_max = min(100, predicted_share + z_score * share_std)
            
            # Calculate area from share
            predicted_area = total_area * predicted_share / 100
            area_min = total_area * share_min / 100
            area_max = total_area * share_max / 100
            
            # Calculate seed demand in quintals (1 quintal = 100 kg)
            seed_demand = predicted_area * self.SEEDING_RATE_KG_PER_HA / 100
            seed_demand_min = area_min * self.SEEDING_RATE_KG_PER_HA / 100
            seed_demand_max = area_max * self.SEEDING_RATE_KG_PER_HA / 100
            
            predictions[variety] = {
                'predicted_share': round(predicted_share, 1),
                'share_range': (round(share_min, 1), round(share_max, 1)),
                'predicted_area_ha': round(predicted_area, 0),
                'area_range_ha': (round(area_min, 0), round(area_max, 0)),
                'seed_demand_qtl': round(seed_demand, 0),
                'seed_demand_range_qtl': (round(seed_demand_min, 0), round(seed_demand_max, 0))
            }
        
        return predictions
    
    def save_model(self, path: str = 'seed_demand_model.pkl'):
        """Save model to file."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'mean_total_area': self.mean_total_area,
            'historical_shares': self.historical_shares,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = 'seed_demand_model.pkl'):
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.mean_total_area = model_data['mean_total_area']
        self.historical_shares = model_data['historical_shares']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {path}")


def train_and_evaluate(excel_path: str = 'Mba BA project.xlsx'):
    """
    Train model and display evaluation metrics.
    """
    print("=" * 60)
    print("SEED DEMAND PREDICTION MODEL - TRAINING")
    print("=" * 60)
    
    # Load and process data
    print("\n📊 Loading and processing data...")
    df = prepare_data_for_modeling(excel_path)
    print(f"   Loaded {len(df)} years of data ({df['Year'].min()}-{df['Year'].max()})")
    
    # Train model
    print("\n🔧 Training models...")
    model = SeedDemandPredictor()
    metrics = model.train(df)
    
    print("\n📈 Training Results:")
    print("-" * 60)
    print(f"{'Variety':<12} {'R² Score':<12} {'MAE':<10} {'Samples':<10} {'Avg Share':<10}")
    print("-" * 60)
    
    for variety, m in metrics.items():
        r2_str = f"{m['r2']:.3f}" if m['r2'] is not None else "N/A"
        mae_str = f"{m['mae']:.2f}%" if m['mae'] is not None else "N/A"
        print(f"{variety:<12} {r2_str:<12} {mae_str:<10} {m['n_samples']:<10} {m['mean_share']:.1f}%")
    
    print("-" * 60)
    print(f"Historical Mean Total Area: {model.mean_total_area:,.0f} hectares")
    
    # Save model
    model.save_model()
    
    return model


if __name__ == "__main__":
    model = train_and_evaluate()
    
    # Test prediction with sample data
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION TEST")
    print("=" * 60)
    
    sample_input = {
        'Max_Temp': 43.0,
        'Min_Temp': 21.0,
        'Pre_Monsoon_Rainfall': 50.0,
        'Monsoon_Rainfall': 200.0,
        'Post_Monsoon_Rainfall': 30.0,
        'Monsoon_Duration': 90
    }
    
    predictions = model.predict(sample_input)
    
    print("\nInput Data:")
    for key, val in sample_input.items():
        print(f"   {key}: {val}")
    
    print("\nPredicted Seed Demand Ranges (in Quintals):")
    print("-" * 70)
    print(f"{'Variety':<12} {'Min Demand':<15} {'Predicted':<15} {'Max Demand':<15}")
    print("-" * 70)
    
    for variety, pred in predictions.items():
        min_d, max_d = pred['seed_demand_range_qtl']
        print(f"{variety:<12} {min_d:>10,.0f}     {pred['seed_demand_qtl']:>10,.0f}     {max_d:>10,.0f}")
    
    print("-" * 70)

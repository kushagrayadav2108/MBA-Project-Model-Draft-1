#!/usr/bin/env python3
"""
Seed Demand Prediction Interface
Interactive CLI to predict seed demand for rice varieties based on meteorological data
"""
import sys
from pathlib import Path
from seed_demand_model import SeedDemandPredictor, train_and_evaluate
from data_processor import prepare_data_for_modeling


def print_header():
    """Print the application header."""
    print("\n" + "=" * 70)
    print("   🌾 SEED DEMAND PREDICTION SYSTEM")
    print("   For Rice Varieties: Pb-1121, Pb-1718, Pb-1885, Pb-1509, Pb-1692, Pb-1847")
    print("=" * 70)


def get_float_input(prompt: str, default: float = None) -> float:
    """Get float input from user with optional default."""
    while True:
        try:
            if default is not None:
                user_input = input(f"   {prompt} [{default}]: ").strip()
                if user_input == '':
                    return default
            else:
                user_input = input(f"   {prompt}: ").strip()
            return float(user_input)
        except ValueError:
            print("   ⚠️  Please enter a valid number.")


def collect_input_data() -> dict:
    """Collect meteorological input data from user."""
    print("\n📋 Enter Meteorological Data for Prediction:")
    print("-" * 50)
    
    data = {
        'Max_Temp': get_float_input("Maximum Temperature (°C)", 43.0),
        'Min_Temp': get_float_input("Minimum Temperature (°C)", 21.0),
        'Pre_Monsoon_Rainfall': get_float_input("Pre-Monsoon Rainfall (mm)", 50.0),
        'Monsoon_Rainfall': get_float_input("Monsoon Rainfall (mm)", 200.0),
        'Post_Monsoon_Rainfall': get_float_input("Post-Monsoon Rainfall (mm)", 30.0),
        'Monsoon_Duration': get_float_input("Monsoon Duration (days)", 90.0)
    }
    
    # Optional: Allow user to specify expected total area
    print("\n   (Optional) Estimated Total Growing Area:")
    area_input = input("   Total Area in Hectares [use historical avg]: ").strip()
    if area_input:
        try:
            data['Total_Area'] = float(area_input)
        except ValueError:
            pass
    
    return data


def display_predictions(predictions: dict, input_data: dict):
    """Display prediction results in a formatted table."""
    print("\n" + "=" * 70)
    print("   📊 SEED DEMAND PREDICTION RESULTS")
    print("=" * 70)
    
    print("\n   Input Parameters:")
    print("   " + "-" * 40)
    print(f"   • Max Temperature:       {input_data['Max_Temp']}°C")
    print(f"   • Min Temperature:       {input_data['Min_Temp']}°C")
    print(f"   • Pre-Monsoon Rainfall:  {input_data['Pre_Monsoon_Rainfall']} mm")
    print(f"   • Monsoon Rainfall:      {input_data['Monsoon_Rainfall']} mm")
    print(f"   • Post-Monsoon Rainfall: {input_data['Post_Monsoon_Rainfall']} mm")
    print(f"   • Monsoon Duration:      {input_data['Monsoon_Duration']} days")
    if 'Total_Area' in input_data:
        print(f"   • Expected Total Area:   {input_data['Total_Area']:,.0f} hectares")
    
    print("\n   " + "=" * 66)
    print("   PREDICTED SEED DEMAND RANGES (in Quintals)")
    print("   " + "=" * 66)
    print(f"   {'Variety':<12} {'Min Demand':>12} {'Predicted':>12} {'Max Demand':>12} {'Share %':>10}")
    print("   " + "-" * 66)
    
    total_min = 0
    total_pred = 0
    total_max = 0
    
    for variety, pred in predictions.items():
        min_d, max_d = pred['seed_demand_range_qtl']
        share = pred['predicted_share']
        
        total_min += min_d
        total_pred += pred['seed_demand_qtl']
        total_max += max_d
        
        # Format variety name for display
        display_name = variety.replace('_', '-')
        
        print(f"   {display_name:<12} {min_d:>12,.0f} {pred['seed_demand_qtl']:>12,.0f} {max_d:>12,.0f} {share:>9.1f}%")
    
    print("   " + "-" * 66)
    print(f"   {'TOTAL':<12} {total_min:>12,.0f} {total_pred:>12,.0f} {total_max:>12,.0f}")
    print("   " + "=" * 66)
    
    # Production planning recommendations
    print("\n   📋 PRODUCTION PLANNING RECOMMENDATIONS:")
    print("   " + "-" * 50)
    print("   To avoid excess/insufficient production:")
    print("   • Plan seed production at the PREDICTED value")
    print("   • Maintain buffer stock for MAX range")
    print("   • MIN range is the conservative estimate")
    print()
    
    return predictions


def save_predictions_to_csv(predictions: dict, input_data: dict, filename: str = 'seed_demand_predictions.csv'):
    """Save predictions to CSV file."""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write input parameters
        writer.writerow(['Input Parameters'])
        for key, val in input_data.items():
            writer.writerow([key, val])
        writer.writerow([])
        
        # Write predictions
        writer.writerow(['Variety', 'Predicted Share %', 'Min Demand (Qtl)', 
                        'Predicted Demand (Qtl)', 'Max Demand (Qtl)',
                        'Min Area (Ha)', 'Predicted Area (Ha)', 'Max Area (Ha)'])
        
        for variety, pred in predictions.items():
            min_d, max_d = pred['seed_demand_range_qtl']
            min_a, max_a = pred['area_range_ha']
            writer.writerow([
                variety.replace('_', '-'),
                pred['predicted_share'],
                min_d,
                pred['seed_demand_qtl'],
                max_d,
                min_a,
                pred['predicted_area_ha'],
                max_a
            ])
    
    print(f"   ✅ Predictions saved to: {filename}")


def run_test_mode(model: SeedDemandPredictor):
    """Run model in test mode with historical data validation."""
    print("\n" + "=" * 70)
    print("   🔬 MODEL VALIDATION - Testing with Historical Data")
    print("=" * 70)
    
    # Load data
    df = prepare_data_for_modeling('Mba BA project.xlsx')
    
    # Test with known 2022 data
    test_years = [2021, 2022]
    
    for year in test_years:
        row = df[df['Year'] == year]
        if len(row) == 0:
            continue
            
        row = row.iloc[0]
        
        # Check if we have meteorological data for this year
        if pd.isna(row['Max_Temp']) or pd.isna(row['Monsoon_Rainfall']):
            continue
        
        print(f"\n   Year {year} - Actual vs Predicted:")
        print("   " + "-" * 50)
        
        input_data = {
            'Max_Temp': row['Max_Temp'],
            'Min_Temp': row['Min_Temp'],
            'Pre_Monsoon_Rainfall': row['Pre_Monsoon_Rainfall'],
            'Monsoon_Rainfall': row['Monsoon_Rainfall'],
            'Post_Monsoon_Rainfall': row['Post_Monsoon_Rainfall'],
            'Monsoon_Duration': row['Monsoon_Duration']
        }
        
        if not pd.isna(row['Total_Area_Hectare']):
            input_data['Total_Area'] = row['Total_Area_Hectare']
        
        predictions = model.predict(input_data)
        
        print(f"   {'Variety':<12} {'Actual Share':>15} {'Predicted Share':>15}")
        print("   " + "-" * 50)
        
        for variety in model.VARIETIES:
            actual_share = row[f'{variety}_Share']
            pred_share = predictions[variety]['predicted_share']
            
            if pd.notna(actual_share):
                diff = pred_share - actual_share
                print(f"   {variety.replace('_', '-'):<12} {actual_share:>14.1f}% {pred_share:>14.1f}% ({diff:+.1f})")
            else:
                print(f"   {variety.replace('_', '-'):<12} {'N/A':>15} {pred_share:>14.1f}%")


def main():
    """Main function to run the prediction interface."""
    print_header()
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode
        model_path = Path('seed_demand_model.pkl')
        if model_path.exists():
            model = SeedDemandPredictor()
            model.load_model(str(model_path))
        else:
            model = train_and_evaluate()
        run_test_mode(model)
        return
    
    # Load or train model
    model_path = Path('seed_demand_model.pkl')
    if model_path.exists():
        print("\n📦 Loading trained model...")
        model = SeedDemandPredictor()
        model.load_model(str(model_path))
    else:
        print("\n🔧 No saved model found. Training new model...")
        model = train_and_evaluate()
    
    # Main prediction loop
    while True:
        print("\n" + "-" * 50)
        print("   Options:")
        print("   1. Make a new prediction")
        print("   2. Retrain model with latest data")
        print("   3. Run validation test")
        print("   4. Exit")
        print("-" * 50)
        
        choice = input("   Enter choice (1-4): ").strip()
        
        if choice == '1':
            input_data = collect_input_data()
            predictions = model.predict(input_data)
            display_predictions(predictions, input_data)
            
            save_choice = input("\n   Save predictions to CSV? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_predictions_to_csv(predictions, input_data)
                
        elif choice == '2':
            print("\n🔄 Retraining model...")
            model = train_and_evaluate()
            
        elif choice == '3':
            run_test_mode(model)
            
        elif choice == '4':
            print("\n   👋 Goodbye! Happy farming!\n")
            break
        else:
            print("   ⚠️  Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    import pandas as pd
    main()

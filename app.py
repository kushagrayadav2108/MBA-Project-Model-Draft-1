"""
Flask API Server for Seed Demand Prediction
Serves the web frontend and handles prediction requests
"""
from flask import Flask, request, jsonify, send_from_directory
import os
from seed_demand_model import SeedDemandPredictor, train_and_evaluate
from pathlib import Path

app = Flask(__name__, static_folder='.')

# Load or train model on startup
model_path = Path('seed_demand_model.pkl')
if model_path.exists():
    print("Loading trained model...")
    model = SeedDemandPredictor()
    model.load_model(str(model_path))
else:
    print("Training new model...")
    model = train_and_evaluate()


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        
        input_data = {
            'Max_Temp': float(data.get('max_temp', 43.0)),
            'Min_Temp': float(data.get('min_temp', 21.0)),
            'Pre_Monsoon_Rainfall': float(data.get('pre_monsoon', 50.0)),
            'Monsoon_Rainfall': float(data.get('monsoon', 200.0)),
            'Post_Monsoon_Rainfall': float(data.get('post_monsoon', 30.0)),
            'Monsoon_Duration': float(data.get('monsoon_duration', 90.0))
        }
        
        if data.get('total_area'):
            input_data['Total_Area'] = float(data['total_area'])
        
        predictions = model.predict(input_data)
        
        # Format response with variety info
        result = {
            'success': True,
            'input': input_data,
            'predictions': {}
        }
        
        for variety, pred in predictions.items():
            variety_info = model.VARIETY_INFO.get(variety, {})
            result['predictions'][variety] = {
                'display_name': variety.replace('_', '-'),
                'duration_category': variety_info.get('category', 'Unknown'),
                'growth_days': variety_info.get('days', 'N/A'),
                'predicted_share': pred['predicted_share'],
                'share_range': pred['share_range'],
                'predicted_area_ha': pred['predicted_area_ha'],
                'area_range_ha': pred['area_range_ha'],
                'seed_demand_qtl': pred['seed_demand_qtl'],
                'seed_demand_range_qtl': pred['seed_demand_range_qtl']
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/variety-info')
def variety_info():
    """Get information about all varieties."""
    return jsonify({
        'varieties': model.VARIETY_INFO,
        'seeding_rate_kg_per_ha': model.SEEDING_RATE_KG_PER_HA,
        'mean_total_area': model.mean_total_area
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🌾 Seed Demand Prediction Server")
    print("=" * 50)
    print("Open http://localhost:8080 in your browser")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', debug=False, port=8080)

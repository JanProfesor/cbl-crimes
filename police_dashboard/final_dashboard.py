from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for data storage
allocation_data = None
processed_data = None
prediction_data = None
geojson_data = None
ward_lookup = None

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATHS = {
    'allocation': 'enhanced_allocation_results_actual.csv',
    'processed': 'processed/final_dataset_residential_burglary.csv',
    'predictions': 'realistic_detailed_predictions.csv',
    'ward_lookup': 'data_preparation/z_old/lsoa_to_ward_lookup_2020.csv',
    'ward_names': 'data_preparation/z_old/Wards_names.csv',
    'geojson': 'data_preparation/z_old/wards_2020_bsc_wgs84.geojson'
}

def load_all_data():
    """Load all required datasets with error handling"""
    global allocation_data, processed_data, prediction_data, geojson_data, ward_lookup
    
    print("🔄 Loading all datasets...")
    
    try:
        # Load allocation data
        if os.path.exists(DATA_PATHS['allocation']):
            allocation_data = pd.read_csv(DATA_PATHS['allocation'])
            print(f"✅ Allocation data loaded: {len(allocation_data)} records")
        else:
            print(f"⚠️ Allocation data not found: {DATA_PATHS['allocation']}")
        
        # Load processed data
        if os.path.exists(DATA_PATHS['processed']):
            processed_data = pd.read_csv(DATA_PATHS['processed'])
            processed_data['date'] = pd.to_datetime(processed_data[['year','month']].assign(day=1))
            print(f"✅ Processed data loaded: {len(processed_data)} records")
        else:
            print(f"⚠️ Processed data not found: {DATA_PATHS['processed']}")
        
        # Load prediction data
        if os.path.exists(DATA_PATHS['predictions']):
            prediction_data = pd.read_csv(DATA_PATHS['predictions'])
            prediction_data['date'] = pd.to_datetime(prediction_data[['year','month']].assign(day=1))
            print(f"✅ Prediction data loaded: {len(prediction_data)} records")
        else:
            print(f"⚠️ Prediction data not found: {DATA_PATHS['predictions']}")
        
        # Load ward lookup
        if os.path.exists(DATA_PATHS['ward_lookup']):
            ward_lookup = pd.read_csv(DATA_PATHS['ward_lookup'])
            ward_lookup = ward_lookup[['WD20CD', 'WD20NM']].drop_duplicates()
            ward_lookup = ward_lookup.rename(columns={'WD20CD': 'ward_code', 'WD20NM': 'ward_name'})
            print(f"✅ Ward lookup loaded: {len(ward_lookup)} wards")
        else:
            print(f"⚠️ Ward lookup not found: {DATA_PATHS['ward_lookup']}")
        
        # Load GeoJSON
        if os.path.exists(DATA_PATHS['geojson']):
            with open(DATA_PATHS['geojson'], 'r') as f:
                geojson_data = json.load(f)
            print(f"✅ GeoJSON loaded")
        else:
            print(f"⚠️ GeoJSON not found: {DATA_PATHS['geojson']}")
        
        # Merge ward names into datasets
        if ward_lookup is not None:
            if processed_data is not None:
                processed_data = processed_data.merge(ward_lookup, on='ward_code', how='left')
            if prediction_data is not None:
                prediction_data = prediction_data.merge(ward_lookup, on='ward_code', how='left')
        
        print("✅ All data loading completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('unified_dashboard.html')

@app.route('/api/summary')
def get_summary():
    """Get overall data summary"""
    try:
        summary = {
            'data_loaded': {
                'allocation': allocation_data is not None,
                'processed': processed_data is not None,
                'predictions': prediction_data is not None,
                'geojson': geojson_data is not None
            },
            'record_counts': {},
            'date_ranges': {},
            'ward_counts': {}
        }
        
        if allocation_data is not None:
            summary['record_counts']['allocation'] = len(allocation_data)
            summary['ward_counts']['allocation'] = allocation_data['ward_code'].nunique()
            if 'year' in allocation_data.columns:
                summary['date_ranges']['allocation'] = {
                    'min_year': int(allocation_data['year'].min()),
                    'max_year': int(allocation_data['year'].max())
                }
        
        if processed_data is not None:
            summary['record_counts']['processed'] = len(processed_data)
            summary['ward_counts']['processed'] = processed_data['ward_code'].nunique()
            summary['date_ranges']['processed'] = {
                'min_date': processed_data['date'].min().isoformat(),
                'max_date': processed_data['date'].max().isoformat()
            }
        
        if prediction_data is not None:
            summary['record_counts']['predictions'] = len(prediction_data)
            summary['ward_counts']['predictions'] = prediction_data['ward_code'].nunique()
            summary['date_ranges']['predictions'] = {
                'min_date': prediction_data['date'].min().isoformat(),
                'max_date': prediction_data['date'].max().isoformat()
            }
        
        return jsonify(summary)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/allocation')
def get_allocation_data():
    """Get allocation data with optional filtering"""
    try:
        if allocation_data is None:
            return jsonify({'error': 'Allocation data not loaded'}), 404
        
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        ward_filter = request.args.get('ward_filter', '')
        
        data = allocation_data.copy()
        
        if year:
            data = data[data['year'] == year]
        if month:
            data = data[data['month'] == month]
        if ward_filter:
            data = data[data['ward_code'].str.contains(ward_filter, case=False, na=False)]
        
        return jsonify({
            'data': data.to_dict('records'),
            'total_records': len(data),
            'filters_applied': {'year': year, 'month': month, 'ward_filter': ward_filter}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get summary statistics for dashboard KPIs"""
    try:
        stats = {}
        
        if processed_data is not None:
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            data = processed_data.copy()
            
            if start_date and end_date:
                data = data[
                    (data['date'] >= pd.to_datetime(start_date)) &
                    (data['date'] <= pd.to_datetime(end_date))
                ]
            
            stats['processed'] = {
                'total_burglaries': int(data['burglary_count'].sum()),
                'avg_burglaries_per_ward': float(data.groupby('ward_code')['burglary_count'].mean().mean()),
                'avg_house_price': float(data['house_price'].mean()),
                'avg_crime_score': float(data['crime_score'].mean()),
                'total_wards': int(data['ward_code'].nunique()),
                'date_range': {
                    'min_date': data['date'].min().isoformat() if len(data) > 0 else None,
                    'max_date': data['date'].max().isoformat() if len(data) > 0 else None
                }
            }
        
        if allocation_data is not None:
            year = request.args.get('year', type=int)
            month = request.args.get('month', type=int)
            
            data = allocation_data.copy()
            if year:
                data = data[data['year'] == year]
            if month:
                data = data[data['month'] == month]
            
            stats['allocation'] = {
                'total_officers': float(data['allocated_officers'].sum()),
                'avg_officers_per_ward': float(data['allocated_officers'].mean()),
                'max_officers': float(data['allocated_officers'].max()),
                'min_officers': float(data['allocated_officers'].min()),
                'total_wards': int(data['ward_code'].nunique()),
                'high_risk_wards': int((data['risk_category'] == 'Critical').sum()),
                'surge_activations': int((data['capacity_multiplier'] > 1.0).sum())
            }
        
        if prediction_data is not None:
            stats['predictions'] = {
                'total_actual_crimes': float(prediction_data['actual'].sum()),
                'total_predicted_crimes': float(prediction_data['pred_ensemble'].sum()),
                'avg_prediction_error': float(prediction_data['abs_error'].mean()),
                'prediction_accuracy': float(100 - prediction_data['abs_pct_error'].mean())
            }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/time_series')
def get_time_series_data():
    """Get time series data for charts"""
    try:
        chart_type = request.args.get('type', 'burglary')
        
        if processed_data is None:
            return jsonify({'error': 'Processed data not available'}), 404
        
        if chart_type == 'burglary':
            ts_data = processed_data.groupby('date')['burglary_count'].mean().reset_index()
            ts_data['date'] = ts_data['date'].dt.strftime('%Y-%m-%d')
            
            return jsonify({
                'data': ts_data.to_dict('records'),
                'title': 'Average Monthly Burglaries',
                'x_label': 'Date',
                'y_label': 'Burglary Count'
            })
        
        elif chart_type == 'house_price':
            ts_data = processed_data.groupby('date')['house_price'].mean().reset_index()
            ts_data['date'] = ts_data['date'].dt.strftime('%Y-%m-%d')
            
            return jsonify({
                'data': ts_data.to_dict('records'),
                'title': 'Average House Prices Over Time',
                'x_label': 'Date',
                'y_label': 'House Price (£)'
            })
        
        return jsonify({'error': 'Invalid chart type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/top_wards')
def get_top_wards():
    """Get top wards data for bar charts"""
    try:
        metric = request.args.get('metric', 'burglary')
        limit = request.args.get('limit', 10, type=int)
        
        if processed_data is None:
            return jsonify({'error': 'Processed data not available'}), 404
        
        if metric == 'burglary':
            top_wards = (
                processed_data.groupby(['ward_code', 'ward_name'])['burglary_count']
                .mean()
                .nlargest(limit)
                .reset_index()
            )
            
            return jsonify({
                'data': top_wards.to_dict('records'),
                'title': f'Top {limit} Wards by Average Burglaries',
                'x_label': 'Ward',
                'y_label': 'Average Burglary Count'
            })
        
        elif metric == 'crime_score':
            top_wards = (
                processed_data.groupby(['ward_code', 'ward_name'])['crime_score']
                .mean()
                .nlargest(limit)
                .reset_index()
            )
            
            return jsonify({
                'data': top_wards.to_dict('records'),
                'title': f'Top {limit} Wards by Crime Score',
                'x_label': 'Ward',
                'y_label': 'Crime Score'
            })
        
        return jsonify({'error': 'Invalid metric'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<ward_code>')
def get_ward_predictions(ward_code):
    """Get prediction data for specific ward"""
    try:
        if prediction_data is None:
            return jsonify({'error': 'Prediction data not available'}), 404
        
        ward_data = prediction_data[prediction_data['ward_code'] == ward_code].copy()
        ward_data = ward_data.sort_values('date')
        ward_data['date'] = ward_data['date'].dt.strftime('%Y-%m-%d')
        
        if len(ward_data) == 0:
            return jsonify({'error': 'No data found for this ward'}), 404
        
        return jsonify({
            'data': ward_data.to_dict('records'),
            'ward_code': ward_code,
            'ward_name': ward_data['ward_name'].iloc[0] if 'ward_name' in ward_data.columns else ward_code,
            'record_count': len(ward_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/map/geojson')
def get_geojson():
    """Get GeoJSON data for map"""
    try:
        if geojson_data is None:
            return jsonify({'error': 'GeoJSON data not available'}), 404
        
        return jsonify(geojson_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/map/choropleth')
def get_choropleth_data():
    """Get data for choropleth map"""
    try:
        data_type = request.args.get('type', 'allocation')
        year = request.args.get('year', type=int)
        month = request.args.get('month', type=int)
        
        if data_type == 'allocation' and allocation_data is not None:
            data = allocation_data.copy()
            if year:
                data = data[data['year'] == year]
            if month:
                data = data[data['month'] == month]
            
            # Group by ward to get single value per ward
            choropleth_data = data.groupby('ward_code').agg({
                'allocated_officers': 'mean',
                'burglary_count': 'mean',
                'risk_category': 'first',
                'adaptive_risk_score': 'mean'
            }).reset_index()
            
            return jsonify({
                'data': choropleth_data.to_dict('records'),
                'value_column': 'allocated_officers',
                'title': 'Officer Allocation by Ward'
            })
        
        elif data_type == 'predictions' and prediction_data is not None:
            data = prediction_data.copy()
            if year:
                data = data[data['year'] == year]
            if month:
                data = data[data['month'] == month]
            
            choropleth_data = data.groupby('ward_code').agg({
                'pred_ensemble': 'mean',
                'actual': 'mean',
                'abs_error': 'mean'
            }).reset_index()
            
            return jsonify({
                'data': choropleth_data.to_dict('records'),
                'value_column': 'pred_ensemble',
                'title': 'Predicted Burglaries by Ward'
            })
        
        return jsonify({'error': 'Invalid data type or data not available'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation')
def get_correlation_data():
    """Get correlation data for scatter plots"""
    try:
        if processed_data is None:
            return jsonify({'error': 'Processed data not available'}), 404
        
        x_var = request.args.get('x', 'crime_score')
        y_var = request.args.get('y', 'house_price')
        sample_size = request.args.get('sample', 1000, type=int)
        
        # Sample data if too large
        data = processed_data.copy()
        if len(data) > sample_size:
            data = data.sample(sample_size, random_state=42)
        
        # Calculate correlation
        correlation = data[x_var].corr(data[y_var])
        
        return jsonify({
            'data': data[[x_var, y_var, 'ward_name']].to_dict('records'),
            'correlation': float(correlation),
            'x_variable': x_var,
            'y_variable': y_var,
            'sample_size': len(data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ward_list')
def get_ward_list():
    """Get list of all wards for dropdowns"""
    try:
        wards = []
        
        if allocation_data is not None:
            allocation_wards = allocation_data[['ward_code']].drop_duplicates()
            if ward_lookup is not None:
                allocation_wards = allocation_wards.merge(ward_lookup, on='ward_code', how='left')
            wards.extend(allocation_wards.to_dict('records'))
        
        # Remove duplicates and sort
        unique_wards = {ward['ward_code']: ward for ward in wards}.values()
        sorted_wards = sorted(unique_wards, key=lambda x: x['ward_code'])
        
        return jsonify({
            'wards': sorted_wards,
            'total_count': len(sorted_wards)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get model performance metrics"""
    try:
        if prediction_data is None:
            return jsonify({'error': 'Prediction data not available'}), 404
        
        # Calculate various performance metrics
        mae = prediction_data['abs_error'].mean()
        rmse = np.sqrt((prediction_data['error'] ** 2).mean())
        mape = prediction_data['abs_pct_error'].mean()
        
        # Calculate R²
        ss_res = ((prediction_data['actual'] - prediction_data['pred_ensemble']) ** 2).sum()
        ss_tot = ((prediction_data['actual'] - prediction_data['actual'].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)
        
        # Error distribution
        error_dist = prediction_data['error_magnitude'].value_counts().to_dict()
        
        return jsonify({
            'metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r_squared': float(r_squared),
                'accuracy': float(100 - mape)
            },
            'error_distribution': error_dist,
            'total_predictions': len(prediction_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Unified London Police Dashboard...")
    print("📂 Loading all required datasets...")
    
    # Try to load all data on startup
    if load_all_data():
        print("✅ All data loaded successfully!")
    else:
        print("⚠️ Some data files missing - dashboard will have limited functionality")
    
    print("🌐 Dashboard will be available at: http://127.0.0.1:5000")
    print("🔧 API endpoints available at: http://127.0.0.1:5000/api/*")
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
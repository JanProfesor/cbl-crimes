from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import requests
from functools import lru_cache

app = Flask(__name__)

# Global variables for data storage
allocation_data = None
prediction_data = None
geojson_cache = None

# ✅ NEW: ArcGIS GeoJSON URL
GEOJSON_URL = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Wards_May_2024_Boundaries_UK_BSC/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"

@lru_cache(maxsize=1)
def fetch_geojson():
    """Fetch and cache GeoJSON data from ArcGIS service"""
    global geojson_cache
    
    if geojson_cache is not None:
        return geojson_cache
    
    try:
        print("🗺️ Fetching GeoJSON from ArcGIS service...")
        response = requests.get(GEOJSON_URL, timeout=30)
        response.raise_for_status()
        
        geojson_data = response.json()
        geojson_cache = geojson_data
        
        print(f"✅ GeoJSON loaded: {len(geojson_data.get('features', []))} features")
        
        # Print sample feature to understand structure
        if geojson_data.get('features'):
            sample_feature = geojson_data['features'][0]
            print(f"📋 Sample feature properties: {list(sample_feature.get('properties', {}).keys())}")
        
        return geojson_data
        
    except requests.RequestException as e:
        print(f"❌ Error fetching GeoJSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing GeoJSON: {e}")
        return None

def safe_read_csv(filepath, **kwargs):
    """Safely read CSV with proper encoding handling"""
    print(f"📁 Attempting to load: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"   Trying encoding: {encoding}")
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            print(f"   ✅ Successfully loaded with {encoding} encoding")
            print(f"   📊 Shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            print(f"   ❌ Failed with {encoding} encoding")
            continue
        except Exception as e:
            print(f"   ❌ Error with {encoding}: {e}")
            continue
    
    print(f"❌ Could not read {filepath} with any supported encoding")
    return None

def load_data():
    """Load available data files"""
    global allocation_data, prediction_data
    
    print("🚀 Loading data files...")
    print("=" * 60)
    
    # Load allocation data (primary file)
    allocation_data = safe_read_csv('enhanced_allocation_results_actual.csv')
    
    # Load prediction data (secondary file)  
    prediction_data = safe_read_csv('realistic_detailed_predictions.csv')
    
    print("=" * 60)
    print("📊 LOADING SUMMARY:")
    
    if allocation_data is not None:
        print(f"✅ Allocation data: {len(allocation_data)} records")
        print(f"   Columns: {list(allocation_data.columns)[:5]}...")
        print(f"   Wards: {allocation_data['ward_code'].nunique()}")
        if 'year' in allocation_data.columns:
            print(f"   Years: {sorted(allocation_data['year'].unique())}")
    else:
        print("❌ Allocation data: Not loaded")
    
    if prediction_data is not None:
        print(f"✅ Prediction data: {len(prediction_data)} records")
        print(f"   Columns: {list(prediction_data.columns)[:5]}...")
        print(f"   Wards: {prediction_data['ward_code'].nunique()}")
        if 'year' in prediction_data.columns:
            print(f"   Years: {sorted(prediction_data['year'].unique())}")
    else:
        print("❌ Prediction data: Not loaded")
    
    return allocation_data is not None or prediction_data is not None

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
                'predictions': prediction_data is not None
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
        
        if prediction_data is not None:
            summary['record_counts']['predictions'] = len(prediction_data)
            summary['ward_counts']['predictions'] = prediction_data['ward_code'].nunique()
            if 'year' in prediction_data.columns:
                summary['date_ranges']['predictions'] = {
                    'min_year': int(prediction_data['year'].min()),
                    'max_year': int(prediction_data['year'].max())
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
        
        if year and 'year' in data.columns:
            data = data[data['year'] == year]
        if month and 'month' in data.columns:
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
        
        if allocation_data is not None:
            year = request.args.get('year', type=int)
            month = request.args.get('month', type=int)
            
            data = allocation_data.copy()
            if year and 'year' in data.columns:
                data = data[data['year'] == year]
            if month and 'month' in data.columns:
                data = data[data['month'] == month]
            
            # Calculate total actual crimes from allocation data
            total_crimes = 0
            if 'actual' in data.columns:
                total_crimes = float(data['actual'].sum())
            elif 'burglary_count' in data.columns:
                total_crimes = float(data['burglary_count'].sum())
            
            stats['allocation'] = {
                'total_officers': float(data['allocated_officers'].sum()) if 'allocated_officers' in data.columns else 0,
                'avg_officers_per_ward': float(data['allocated_officers'].mean()) if 'allocated_officers' in data.columns else 0,
                'max_officers': float(data['allocated_officers'].max()) if 'allocated_officers' in data.columns else 0,
                'min_officers': float(data['allocated_officers'].min()) if 'allocated_officers' in data.columns else 0,
                'total_wards': int(data['ward_code'].nunique()),
                'high_risk_wards': int((data['risk_category'] == 'Critical').sum()) if 'risk_category' in data.columns else 0,
                'surge_activations': int((data['capacity_multiplier'] > 1.0).sum()) if 'capacity_multiplier' in data.columns else 0,
                'total_crimes': total_crimes  # ✅ NEW: Add total crimes calculation
            }
        
        if prediction_data is not None:
            # Also calculate from prediction data for backup
            prediction_crimes = 0
            if 'actual' in prediction_data.columns:
                prediction_crimes = float(prediction_data['actual'].sum())
            
            stats['predictions'] = {
                'total_actual_crimes': prediction_crimes,
                'total_predicted_crimes': float(prediction_data['pred_ensemble'].sum()) if 'pred_ensemble' in prediction_data.columns else 0,
                'avg_prediction_error': float(prediction_data['abs_error'].mean()) if 'abs_error' in prediction_data.columns else 0,
                'prediction_accuracy': float(100 - prediction_data['abs_pct_error'].mean()) if 'abs_pct_error' in prediction_data.columns else 0
            }
            
            # ✅ NEW: Use prediction data for total crimes if allocation doesn't have it
            if 'allocation' not in stats or stats['allocation']['total_crimes'] == 0:
                if 'allocation' not in stats:
                    stats['allocation'] = {}
                stats['allocation']['total_crimes'] = prediction_crimes
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ward_list')
def get_ward_list():
    """Get list of all wards for dropdowns"""
    try:
        wards = []
        
        if allocation_data is not None:
            allocation_wards = allocation_data[['ward_code']].drop_duplicates()
            wards.extend([{'ward_code': row['ward_code'], 'ward_name': row['ward_code']} 
                         for _, row in allocation_wards.iterrows()])
        
        # Remove duplicates and sort
        unique_wards = {ward['ward_code']: ward for ward in wards}.values()
        sorted_wards = sorted(unique_wards, key=lambda x: x['ward_code'])
        
        return jsonify({
            'wards': sorted_wards,
            'total_count': len(sorted_wards)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<ward_code>')
def get_ward_predictions(ward_code):
    """Get prediction data for specific ward"""
    try:
        if prediction_data is None:
            return jsonify({'error': 'Prediction data not available'}), 404
        
        ward_data = prediction_data[prediction_data['ward_code'] == ward_code].copy()
        
        if len(ward_data) == 0:
            return jsonify({'error': 'No data found for this ward'}), 404
        
        # Sort by date if possible
        if 'year' in ward_data.columns and 'month' in ward_data.columns:
            ward_data = ward_data.sort_values(['year', 'month'])
            # Create date column
            ward_data['date'] = pd.to_datetime(ward_data[['year','month']].assign(day=1)).dt.strftime('%Y-%m-%d')
        
        return jsonify({
            'data': ward_data.to_dict('records'),
            'ward_code': ward_code,
            'ward_name': ward_code,  # Use code as name for now
            'record_count': len(ward_data)
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
        data = prediction_data
        
        # Check if required columns exist
        required_cols = ['actual', 'pred_ensemble', 'error', 'abs_error', 'abs_pct_error']
        available_cols = [col for col in required_cols if col in data.columns]
        
        if not available_cols:
            return jsonify({'error': 'No prediction columns found'}), 404
        
        metrics = {}
        
        if 'abs_error' in data.columns:
            metrics['mae'] = float(data['abs_error'].mean())
        
        if 'error' in data.columns:
            rmse = np.sqrt((data['error'] ** 2).mean())
            metrics['rmse'] = float(rmse)
        
        if 'abs_pct_error' in data.columns:
            metrics['mape'] = float(data['abs_pct_error'].mean())
            metrics['accuracy'] = float(100 - data['abs_pct_error'].mean())
        
        if 'actual' in data.columns and 'pred_ensemble' in data.columns:
            # Calculate R²
            ss_res = ((data['actual'] - data['pred_ensemble']) ** 2).sum()
            ss_tot = ((data['actual'] - data['actual'].mean()) ** 2).sum()
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            metrics['r_squared'] = float(r_squared)
        
        # Error distribution
        error_dist = {}
        if 'error_magnitude' in data.columns:
            error_dist = data['error_magnitude'].value_counts().to_dict()
        
        return jsonify({
            'metrics': metrics,
            'error_distribution': error_dist,
            'total_predictions': len(data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/time_series')
def get_time_series_data():
    """Get time series data for charts"""
    try:
        chart_type = request.args.get('type', 'burglary')
        
        # Use allocation data as primary source
        if allocation_data is None:
            return jsonify({'error': 'No data available'}), 404
        
        data = allocation_data.copy()
        
        if chart_type == 'burglary' and 'burglary_count' in data.columns:
            # Group by date if possible
            if 'year' in data.columns and 'month' in data.columns:
                data['date'] = pd.to_datetime(data[['year','month']].assign(day=1))
                ts_data = data.groupby('date')['burglary_count'].mean().reset_index()
                ts_data['date'] = ts_data['date'].dt.strftime('%Y-%m-%d')
                
                return jsonify({
                    'data': ts_data.to_dict('records'),
                    'title': 'Average Monthly Burglaries',
                    'x_label': 'Date',
                    'y_label': 'Burglary Count'
                })
        
        return jsonify({'error': 'Chart data not available'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/top_wards')
def get_top_wards():
    """Get top wards data for bar charts"""
    try:
        metric = request.args.get('metric', 'burglary')
        limit = request.args.get('limit', 10, type=int)
        
        if allocation_data is None:
            return jsonify({'error': 'No data available'}), 404
        
        data = allocation_data.copy()
        
        if metric == 'burglary' and 'burglary_count' in data.columns:
            top_wards = (
                data.groupby('ward_code')['burglary_count']
                .mean()
                .nlargest(limit)
                .reset_index()
            )
            top_wards['ward_name'] = top_wards['ward_code']  # Use code as name
            
            return jsonify({
                'data': top_wards.to_dict('records'),
                'title': f'Top {limit} Wards by Average Burglaries',
                'x_label': 'Ward',
                'y_label': 'Average Burglary Count'
            })
        
        return jsonify({'error': 'Invalid metric or data not available'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check what's available"""
    debug_data = {
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'data_loaded': {
            'allocation': allocation_data is not None,
            'predictions': prediction_data is not None,
            'geojson_cache': geojson_cache is not None
        }
    }
    
    if allocation_data is not None:
        debug_data['allocation_info'] = {
            'shape': allocation_data.shape,
            'columns': list(allocation_data.columns),
            'sample_data': allocation_data.head(2).to_dict('records'),
            'year_range': [int(allocation_data['year'].min()), int(allocation_data['year'].max())] if 'year' in allocation_data.columns else 'No year column',
            'month_range': sorted(allocation_data['month'].unique().tolist()) if 'month' in allocation_data.columns else 'No month column',
            'unique_wards': allocation_data['ward_code'].nunique() if 'ward_code' in allocation_data.columns else 'No ward_code column'
        }
    
    if prediction_data is not None:
        debug_data['prediction_info'] = {
            'shape': prediction_data.shape,
            'columns': list(prediction_data.columns),
            'sample_data': prediction_data.head(2).to_dict('records'),
            'year_range': [int(prediction_data['year'].min()), int(prediction_data['year'].max())] if 'year' in prediction_data.columns else 'No year column',
            'month_range': sorted(prediction_data['month'].unique().tolist()) if 'month' in prediction_data.columns else 'No month column'
        }
    
    # Test GeoJSON loading
    try:
        geojson_test = fetch_geojson()
        if geojson_test:
            debug_data['geojson_info'] = {
                'features_count': len(geojson_test.get('features', [])),
                'sample_properties': list(geojson_test['features'][0].get('properties', {}).keys()) if geojson_test.get('features') else 'No features'
            }
        else:
            debug_data['geojson_info'] = 'Failed to load GeoJSON'
    except Exception as e:
        debug_data['geojson_info'] = f'GeoJSON error: {str(e)}'
    
    return jsonify(debug_data)

if __name__ == '__main__':
    print("🚀 Starting Simplified London Police Dashboard...")
    print("📂 Looking for data files in current directory...")
    print("=" * 60)
    
    # List available files
    print("📁 Files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"   ✅ {file}")
    
    print("=" * 60)
    
    # Try to load data on startup
    if load_data():
        print("✅ Data loaded successfully!")
    else:
        print("⚠️ No data files found - dashboard will have limited functionality")
        print("📋 Expected files:")
        print("   - enhanced_allocation_results_actual.csv")
        print("   - realistic_detailed_predictions.csv")
    
    print("=" * 60)
    print("🌐 Dashboard will be available at: http://127.0.0.1:5000")
    print("🔧 Debug info available at: http://127.0.0.1:5000/api/debug")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
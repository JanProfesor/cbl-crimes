# app.py - Improved Flask application with better debugging
from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os

app = Flask(__name__)

# Global variable to store data
allocation_data = None
csv_info = {}

def load_csv_data():
    """Load the CSV data into memory with better error handling"""
    global allocation_data, csv_info
    
    try:
        print("="*50)
        print("🔍 LOOKING FOR ENHANCED ALLOCATION CSV...")
        print("="*50)
        
        # Look for CSV files in the current directory
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        
        all_files = os.listdir('.')
        csv_files = [f for f in all_files if f.endswith('.csv')]
        
        print(f"All files in directory: {all_files}")
        print(f"CSV files found: {csv_files}")
        
        # Priority order for CSV files - look for the main enhanced allocation file
        preferred_files = [
            'enhanced_allocation_results.csv'
        ]
        
        csv_file = None
        
        # Try to find preferred files first
        for preferred in preferred_files:
            if preferred in csv_files:
                # Quick check if this file has the right columns
                try:
                    test_df = pd.read_csv(preferred, nrows=1)
                    if 'allocated_officers' in test_df.columns and 'ward_code' in test_df.columns:
                        csv_file = preferred
                        print(f"✅ Found preferred file with correct columns: {csv_file}")
                        break
                    else:
                        print(f"⏭️  Skipping {preferred} - missing required columns")
                except:
                    print(f"⏭️  Skipping {preferred} - cannot read file")
        
        # If no preferred file found, look for files with required columns
        if csv_file is None:
            for file in csv_files:
                # Skip known problematic files
                if any(skip in file.lower() for skip in ['special_operations', 'holdout', 'test_predictions', 'metrics']):
                    print(f"⏭️  Skipping {file} - not an allocation file")
                    continue
                
                try:
                    # Test if file has required columns
                    test_df = pd.read_csv(file, nrows=1)
                    if 'allocated_officers' in test_df.columns and 'ward_code' in test_df.columns:
                        csv_file = file
                        print(f"📂 Found suitable file: {csv_file}")
                        break
                    else:
                        print(f"⏭️  Skipping {file} - columns: {list(test_df.columns)[:5]}...")
                except Exception as e:
                    print(f"⏭️  Cannot read {file}: {e}")
        
        if csv_file is None:
            print("❌ NO SUITABLE ALLOCATION FILE FOUND!")
            print("Looking for files with 'allocated_officers' and 'ward_code' columns")
            print("Available CSV files and their first few columns:")
            for file in csv_files:
                try:
                    test_df = pd.read_csv(file, nrows=1)
                    print(f"  - {file}: {list(test_df.columns)[:5]}...")
                except:
                    print(f"  - {file}: Cannot read")
            return False
        
        print(f"📂 Loading data from: {csv_file}")
        
        # Read CSV with error handling
        allocation_data = pd.read_csv(csv_file)
        
        print(f"✅ CSV loaded successfully!")
        print(f"📊 Shape: {allocation_data.shape}")
        print(f"📋 Columns: {list(allocation_data.columns)}")
        print(f"🔢 Total rows: {len(allocation_data)}")
        
        # Your file already has the perfect structure! No column mapping needed.
        # Just verify the required columns exist
        required_columns = ['ward_code', 'allocated_officers', 'year', 'month']
        missing_columns = [col for col in required_columns if col not in allocation_data.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(allocation_data.columns)}")
            return False
        
        print("✅ All required columns found!")
        
        # Verify data types
        allocation_data['allocated_officers'] = pd.to_numeric(allocation_data['allocated_officers'], errors='coerce')
        allocation_data['year'] = pd.to_numeric(allocation_data['year'], errors='coerce')
        allocation_data['month'] = pd.to_numeric(allocation_data['month'], errors='coerce')
        
        # Clean the data - remove any rows with missing critical data
        before_cleaning = len(allocation_data)
        allocation_data = allocation_data.dropna(subset=['ward_code', 'allocated_officers', 'year', 'month'])
        after_cleaning = len(allocation_data)
        
        if before_cleaning != after_cleaning:
            print(f"🧹 Cleaned data: removed {before_cleaning - after_cleaning} rows with missing data")
        
        # Clean the data
        print("\n🧹 CLEANING DATA...")
        original_count = len(allocation_data)
        
        # Remove rows with missing ward_code or allocated_officers
        allocation_data = allocation_data.dropna(subset=['ward_code', 'allocated_officers'])
        
        # Convert allocated_officers to numeric
        allocation_data['allocated_officers'] = pd.to_numeric(allocation_data['allocated_officers'], errors='coerce')
        allocation_data = allocation_data.dropna(subset=['allocated_officers'])
        
        print(f"📊 Rows after cleaning: {len(allocation_data)} (removed {original_count - len(allocation_data)})")
        
        # Show sample data
        print("\n📋 SAMPLE DATA (first 3 rows):")
        sample_columns = ['ward_code', 'year', 'month', 'allocated_officers', 'burglary_count', 'risk_category', 'season']
        available_sample_columns = [col for col in sample_columns if col in allocation_data.columns]
        print(allocation_data[available_sample_columns].head(3).to_string())
        
        # Store info for API
        csv_info = {
            'filename': csv_file,
            'total_rows': len(allocation_data),
            'columns': list(allocation_data.columns),
            'sample_data': allocation_data.head(3).to_dict('records')
        }
        
        # Check for year/month columns
        if 'year' in allocation_data.columns and 'month' in allocation_data.columns:
            years = sorted(allocation_data['year'].unique())
            months = sorted(allocation_data['month'].unique())
            print(f"📅 Years available: {years}")
            print(f"📅 Months available: {months}")
        else:
            print("⚠️  No year/month columns found - time filtering won't work")
        
        print("="*50)
        print("✅ DATA LOADING COMPLETE!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR LOADING CSV: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def dashboard():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get all allocation data"""
    global allocation_data
    
    print("📡 API request received: /api/data")
    
    if allocation_data is None:
        print("🔄 Data not loaded, attempting to load...")
        if not load_csv_data():
            error_msg = 'Failed to load CSV data. Check console for details.'
            print(f"❌ Returning error: {error_msg}")
            return jsonify({'error': error_msg}), 500
    
    try:
        # Convert DataFrame to JSON
        data_json = allocation_data.to_dict('records')
        
        response = {
            'data': data_json,
            'total_records': len(data_json),
            'columns': list(allocation_data.columns),
            'csv_info': csv_info
        }
        
        print(f"✅ Returning {len(data_json)} records")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Error converting data to JSON: {str(e)}'
        print(f"❌ API Error: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/data/<int:year>/<int:month>')
def get_data_by_period(year, month):
    """API endpoint to get data for specific year/month"""
    global allocation_data
    
    print(f"📡 API request received: /api/data/{year}/{month}")
    
    if allocation_data is None:
        if not load_csv_data():
            return jsonify({'error': 'Failed to load data'}), 500
    
    try:
        # Filter data by year and month
        filtered_data = allocation_data[
            (allocation_data['year'] == year) & 
            (allocation_data['month'] == month)
        ]
        
        data_json = filtered_data.to_dict('records')
        
        response = {
            'data': data_json,
            'year': year,
            'month': month,
            'total_records': len(data_json)
        }
        
        print(f"✅ Returning {len(data_json)} records for {year}-{month}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Error filtering data: {str(e)}'
        print(f"❌ API Error: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/summary')
def get_summary():
    """API endpoint to get data summary"""
    global allocation_data
    
    print("📡 API request received: /api/summary")
    
    if allocation_data is None:
        if not load_csv_data():
            return jsonify({'error': 'Failed to load data'}), 500
    
    try:
        summary = {
            'years': sorted(allocation_data['year'].unique().tolist()) if 'year' in allocation_data.columns else [],
            'months': sorted(allocation_data['month'].unique().tolist()) if 'month' in allocation_data.columns else [],
            'wards': sorted(allocation_data['ward_code'].unique().tolist()),
            'total_records': len(allocation_data),
            'columns': list(allocation_data.columns)
        }
        
        if 'year' in allocation_data.columns:
            summary['date_range'] = {
                'min_year': int(allocation_data['year'].min()),
                'max_year': int(allocation_data['year'].max()),
            }
        
        print(f"✅ Returning summary: {len(summary['wards'])} wards, {len(summary['years'])} years")
        return jsonify(summary)
        
    except Exception as e:
        error_msg = f'Error generating summary: {str(e)}'
        print(f"❌ API Error: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check what's happening"""
    global allocation_data, csv_info
    
    debug_data = {
        'data_loaded': allocation_data is not None,
        'csv_info': csv_info,
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'csv_files': [f for f in os.listdir('.') if f.endswith('.csv')]
    }
    
    if allocation_data is not None:
        debug_data['data_shape'] = allocation_data.shape
        debug_data['columns'] = list(allocation_data.columns)
        debug_data['sample_data'] = allocation_data.head(2).to_dict('records')
    
    return jsonify(debug_data)

if __name__ == '__main__':
    print("🚀 Starting London Police Dashboard...")
    print("📂 Make sure your CSV file is in the same directory as this script")
    print("🌐 Dashboard will be available at: http://127.0.0.1:5000")
    print("🔧 Debug info available at: http://127.0.0.1:5000/api/debug")
    
    # Try to load data on startup
    load_csv_data()
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
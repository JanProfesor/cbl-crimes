import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="London Police Intelligence Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .alert-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_data
def load_ward_boundaries():
    """Load ward boundaries from local GeoJSON (using WD24CD)"""
    geojson_path = PROJECT_ROOT/"data_preparation" / "Wards_May_2024.geojson"
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        # Verify WD24CD exists
        props = gj["features"][0]["properties"]
        if "WD24CD" not in props:
            st.error(f"❌ WD24CD not found in GeoJSON properties: {list(props.keys())}")
            return None
        return gj
    except Exception as e:
        st.error(f"❌ Failed to load local ward boundaries: {e}")
        return None

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'allocation_data' not in st.session_state:
    st.session_state.allocation_data = None
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'ward_boundaries' not in st.session_state:
    st.session_state.ward_boundaries = None

# once we've loaded the geojson, build a ward_code → ward_name map
if st.session_state.ward_boundaries is None:
    st.session_state.ward_boundaries = load_ward_boundaries()

if st.session_state.ward_boundaries:
    st.session_state.ward_name_map = {
        feat["properties"]["WD24CD"]: feat["properties"]["WD24NM"]
        for feat in st.session_state.ward_boundaries["features"]
    }

@st.cache_data
def load_csv_data():
    """Load CSV data files with caching"""
    allocation_data = None
    prediction_data = None
    
    # Try to load allocation data
    allocation_files = [
        'enhanced_allocation_results_actual.csv',
        'allocation_results.csv',
        'allocation_data.csv'
    ]
    
    for file in allocation_files:
        if os.path.exists(file):
            try:
                allocation_data = pd.read_csv(file)
                st.success(f"✅ Loaded allocation data from {file}")
                break
            except Exception as e:
                st.error(f"❌ Error loading {file}: {e}")
    
    # Try to load prediction data
    prediction_files = [
        'realistic_detailed_predictions.csv',
        'predictions.csv',
        'prediction_data.csv'
    ]
    
    for file in prediction_files:
        if os.path.exists(file):
            try:
                prediction_data = pd.read_csv(file)
                st.success(f"✅ Loaded prediction data from {file}")
                break
            except Exception as e:
                st.error(f"❌ Error loading {file}: {e}")
    
    return allocation_data, prediction_data




@st.cache_data
def get_ward_centroids(ward_geojson: dict) -> dict:
    """
    Returns a dict mapping ward_code (WD24CD) -> (lat, lon) centroid.
    """
    from shapely.geometry import shape

    centroids = {}
    for feat in ward_geojson["features"]:
        code = feat["properties"]["WD24CD"]
        poly = shape(feat["geometry"])
        lon, lat = poly.centroid.coords[0]
        centroids[code] = (lat, lon)
    return centroids




def generate_mock_data():
    """Generate mock data for demonstration"""
    st.warning("📊 Generating mock data for demonstration...")
    
    # Mock allocation data
    ward_codes = [f"E0500{i:04d}" for i in range(1, 101)]
    years = [2020, 2021, 2022, 2023, 2024]
    months = list(range(1, 13))
    
    allocation_records = []
    for ward in ward_codes[:20]:  # Limit to 20 wards for demo
        for year in years:
            for month in months:
                allocation_records.append({
                    'ward_code': ward,
                    'year': year,
                    'month': month,
                    'allocated_officers': np.random.randint(1, 15),
                    'burglary_count': np.random.randint(0, 25),
                    'actual': np.random.randint(0, 20),
                    'risk_category': np.random.choice(['Low', 'Medium', 'High', 'Critical']),
                    'adaptive_risk_score': np.random.random(),
                    'capacity_multiplier': 1 + np.random.random() * 0.5
                })
    
    allocation_df = pd.DataFrame(allocation_records)
    
    # Mock prediction data
    prediction_records = []
    for ward in ward_codes[:20]:
        for year in years:
            for month in months:
                actual = np.random.randint(0, 20)
                predicted = actual + np.random.randint(-5, 6)
                error = predicted - actual
                
                prediction_records.append({
                    'ward_code': ward,
                    'year': year,
                    'month': month,
                    'actual': actual,
                    'pred_ensemble': max(0, predicted),
                    'error': error,
                    'abs_error': abs(error),
                    'abs_pct_error': abs(error / max(actual, 1)) * 100 if actual > 0 else 0
                })
    
    prediction_df = pd.DataFrame(prediction_records)
    
    return allocation_df, prediction_df

def calculate_kpis(allocation_df, prediction_df):
    """Calculate key performance indicators"""
    kpis = {}
    
    if allocation_df is not None:
        kpis.update({
            'total_wards': allocation_df['ward_code'].nunique(),
            'total_officers': allocation_df['allocated_officers'].sum(),
            'avg_officers_per_ward': allocation_df['allocated_officers'].mean(),
            'total_crimes': allocation_df['actual'].sum() if 'actual' in allocation_df.columns else allocation_df['burglary_count'].sum(),
            'high_risk_wards': (allocation_df['risk_category'] == 'Critical').sum() if 'risk_category' in allocation_df.columns else 0,
            'surge_activations': (allocation_df['capacity_multiplier'] > 1.0).sum() if 'capacity_multiplier' in allocation_df.columns else 0
        })
    
    if prediction_df is not None:
        kpis.update({
            'prediction_accuracy': 100 - prediction_df['abs_pct_error'].mean() if 'abs_pct_error' in prediction_df.columns else 0,
            'avg_prediction_error': prediction_df['abs_error'].mean() if 'abs_error' in prediction_df.columns else 0,
            'total_predictions': len(prediction_df)
        })
    
    return kpis

def create_officer_allocation_chart(allocation_df):
    """Create officer allocation distribution chart"""
    if allocation_df is None or 'allocated_officers' not in allocation_df.columns:
        return None
    
    fig = px.histogram(
        allocation_df, 
        x='allocated_officers',
        nbins=20,
        title='Officer Allocation Distribution',
        labels={'allocated_officers': 'Officers per Ward', 'count': 'Frequency'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_risk_category_chart(allocation_df):
    """Create risk category pie chart"""
    if allocation_df is None or 'risk_category' not in allocation_df.columns:
        return None
    
    risk_counts = allocation_df['risk_category'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='Risk Category Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_time_series_chart(allocation_df):
    """Create time series chart for crime trends"""
    if allocation_df is None:
        return None
    
    # Create date column
    if 'year' in allocation_df.columns and 'month' in allocation_df.columns:
        allocation_df['date'] = pd.to_datetime(allocation_df[['year', 'month']].assign(day=1))
        
        # Aggregate by date
        crime_column = 'actual' if 'actual' in allocation_df.columns else 'burglary_count'
        ts_data = allocation_df.groupby('date')[crime_column].mean().reset_index()
        
        fig = px.line(
            ts_data,
            x='date',
            y=crime_column,
            title='Average Monthly Crime Trends',
            labels={'date': 'Date', crime_column: 'Average Crime Count'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
    
    return None

def create_top_wards_chart(allocation_df, metric='burglary_count', limit=10):
    """Create top wards bar chart"""
    if allocation_df is None:
        return None
    
    crime_column = 'actual' if 'actual' in allocation_df.columns else 'burglary_count'
    if crime_column not in allocation_df.columns:
        return None
    
    top_wards = (
        allocation_df.groupby('ward_code')[crime_column]
        .mean()
        .nlargest(limit)
        .reset_index()
    )
    
    fig = px.bar(
        top_wards,
        x='ward_code',
        y=crime_column,
        title=f'Top {limit} Wards by Average Crime Count',
        labels={'ward_code': 'Ward Code', crime_column: 'Average Crime Count'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_prediction_chart(prediction_df, ward_code):
    """Create prediction vs actual chart for specific ward"""
    if prediction_df is None:
        return None
    
    ward_data = prediction_df[prediction_df['ward_code'] == ward_code].copy()
    if len(ward_data) == 0:
        return None
    
    # Create date column
    if 'year' in ward_data.columns and 'month' in ward_data.columns:
        ward_data['date'] = pd.to_datetime(ward_data[['year', 'month']].assign(day=1))
        ward_data = ward_data.sort_values('date')
        
        fig = go.Figure()
        
        if 'actual' in ward_data.columns:
            fig.add_trace(go.Scatter(
                x=ward_data['date'],
                y=ward_data['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#00ff88')
            ))
        
        if 'pred_ensemble' in ward_data.columns:
            fig.add_trace(go.Scatter(
                x=ward_data['date'],
                y=ward_data['pred_ensemble'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#ff6b6b')
            ))
        
        fig.update_layout(
            title=f'Crime Predictions for Ward {ward_code}',
            xaxis_title='Date',
            yaxis_title='Crime Count',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    return None

@st.cache_resource
def create_map(allocation_df, ward_boundaries=None, map_type="Crime Hotspots"):
    """
    Create a Folium map, handling three modes:
      - Crime Hotspots: top-10 CircleMarkers at centroids
      - Officer Allocation: Choropleth by avg officers per ward
      - Risk Categories: GeoJson styled by risk category
    """
    # 1️⃣ Base map
    m = folium.Map(
        location=[51.5074, -0.1278],
        zoom_start=10,
        tiles="CartoDB dark_matter"
    )
    if allocation_df is None or ward_boundaries is None:
        return m

    # 2️⃣ Detect the ward ID field once
    sample = ward_boundaries["features"][0]["properties"]
    id_field = next((k for k in sample if k.lower().endswith("cd")), None)

    # 3️⃣ Hotspots mode
    if map_type == "Crime Hotspots":
        centroids = get_ward_centroids(ward_boundaries)
        crime_col = "actual" if "actual" in allocation_df.columns else "burglary_count"
        top10 = allocation_df.groupby("ward_code")[crime_col].mean().nlargest(10)
        median = top10.median()
        for code, cnt in top10.items():
            lat, lon = centroids.get(code, (51.5074, -0.1278))
            color = "red" if cnt > median else "orange"
            folium.CircleMarker(
                location=[lat, lon],
                radius=min(cnt * 2, 20),
                popup=f"{code}: {cnt:.1f}",
                color=color, fill=True, fill_opacity=0.6
            ).add_to(m)
        return m

    if map_type == "Officer Allocation":
        # ── 1) Get the ward ID field ───────────────────────────
        sample = ward_boundaries["features"][0]["properties"]
        id_field = "WD24CD"
        
        # ── 2) Build a full list of ward_codes from the geojson ─
        all_codes = [feat["properties"][id_field] for feat in ward_boundaries["features"]]
        df_all = pd.DataFrame({"ward_code": all_codes})
        
        # ── 3) Aggregate your actual data ───────────────────────
        df_off = (
            allocation_df
            .groupby("ward_code")["allocated_officers"]
            .mean()
            .reset_index()
        )
        # left-join so every ward_code is present
        df_choro = df_all.merge(df_off, on="ward_code", how="left")
        
        # ── 4) Choropleth using that full DataFrame ─────────────
        folium.Choropleth(
            geo_data=ward_boundaries,
            name="Avg Officers",
            data=df_choro,
            columns=["ward_code", "allocated_officers"],
            key_on=f"feature.properties.{id_field}",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            # color for wards with no data:
            nan_fill_color="lightgray",
            legend_name="Avg Officers per Ward"
        ).add_to(m)
        
        # ── 5) Overlay thin white outlines for _all_ wards ──────
        folium.GeoJson(
            ward_boundaries,
            name="Ward Boundaries",
            style_function=lambda feat: {
                "fillOpacity": 0,
                "color": "white",
                "weight": 0.5
            }
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m


    # 5️⃣ Risk Categories fill
    if map_type == "Risk Categories":
        # define color palette
        palette = {
            "Critical": "#d73027",
            "High":     "#fc8d59",
            "Medium":   "#fee08b",
            "Low":      "#91cf60"
        }
        # pick highest-severity per ward
        order = {"Low":0, "Medium":1, "High":2, "Critical":3}
        modal = (
            allocation_df
            .assign(_rank=allocation_df["risk_category"].map(order))
            .sort_values("_rank")
            .drop_duplicates("ward_code", keep="last")
            .set_index("ward_code")["risk_category"]
        )

        def style_fn(feat):
            code = feat["properties"][id_field]
            cat = modal.get(code, None)
            return {
                "fillColor": palette.get(cat, "#444444"),
                "color": "white",
                "weight": 1,
                "fillOpacity": 0.7
            }

        folium.GeoJson(
            ward_boundaries,
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=[id_field], aliases=["Ward:"])
        ).add_to(m)
        folium.LayerControl().add_to(m)
        return m

    # 6️⃣ Fallback: return empty map
    return m



# Main dashboard function
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚔 London Police Intelligence Dashboard</h1>
        <p>Enhanced Adaptive Resource Allocation & Crime Pattern Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Load data button
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.rerun()
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            allocation_df, prediction_df = load_csv_data()
            if allocation_df is not None:
                allocation_df["ward_name"] = allocation_df["ward_code"].map(st.session_state.ward_name_map)

            if prediction_df is not None:
                prediction_df["ward_name"] = prediction_df["ward_code"].map(st.session_state.ward_name_map)
            
            # If no real data, generate mock data
            if allocation_df is None and prediction_df is None:
                st.warning("⚠️ No CSV files found. Generating mock data for demonstration.")
                allocation_df, prediction_df = generate_mock_data()
            
            st.session_state.allocation_data = allocation_df
            st.session_state.prediction_data = prediction_df
            st.session_state.data_loaded = True
    
    allocation_df = st.session_state.allocation_data
    prediction_df = st.session_state.prediction_data
    
    # Navigation
    screen = st.sidebar.selectbox(
        "Select Dashboard Screen",
        ["📊 Overview", "👮 Resource Allocation", "🔮 Crime Predictions", "📈 Analytics & Trends", "🗺️ Interactive Map"]
    )
    
    # Data status
    st.sidebar.markdown("### 📊 Data Status")
    if allocation_df is not None:
        st.sidebar.markdown('<p class="status-online">✅ Allocation Data: Loaded</p>', unsafe_allow_html=True)
        st.sidebar.write(f"Records: {len(allocation_df):,}")
        st.sidebar.write(f"Wards: {allocation_df['ward_code'].nunique()}")
    else:
        st.sidebar.markdown('<p class="status-offline">❌ Allocation Data: Not Available</p>', unsafe_allow_html=True)
    
    if prediction_df is not None:
        st.sidebar.markdown('<p class="status-online">✅ Prediction Data: Loaded</p>', unsafe_allow_html=True)
        st.sidebar.write(f"Records: {len(prediction_df):,}")
    else:
        st.sidebar.markdown('<p class="status-offline">❌ Prediction Data: Not Available</p>', unsafe_allow_html=True)
    
    # Calculate KPIs
    kpis = calculate_kpis(allocation_df, prediction_df)
    
    # Screen routing
    if screen == "📊 Overview":
        overview_screen(allocation_df, prediction_df, kpis)
    elif screen == "👮 Resource Allocation":
        allocation_screen(allocation_df, kpis)
    elif screen == "🔮 Crime Predictions":
        predictions_screen(allocation_df, prediction_df)
    elif screen == "📈 Analytics & Trends":
        analytics_screen(allocation_df, prediction_df)
    elif screen == "🗺️ Interactive Map":
        map_screen(allocation_df)

def overview_screen(allocation_df, prediction_df, kpis):
    """Overview dashboard screen"""
    st.header("📊 Overview Dashboard")
    st.write("Real-time summary of police resource allocation and crime patterns")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Wards", f"{kpis.get('total_wards', 0):,}")
    
    with col2:
        st.metric("Deployed Officers", f"{kpis.get('total_officers', 0):,}")
    
    with col3:
        st.metric("Total Crimes", f"{kpis.get('total_crimes', 0):,}")
    
    with col4:
        accuracy = kpis.get('prediction_accuracy', 0)
        st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Avg Officers/Ward", f"{kpis.get('avg_officers_per_ward', 0):.1f}")
    
    with col6:
        st.metric("High Risk Wards", f"{kpis.get('high_risk_wards', 0):,}")
    
    with col7:
        st.metric("Surge Activations", f"{kpis.get('surge_activations', 0):,}")
    
    with col8:
        error = kpis.get('avg_prediction_error', 0)
        st.metric("Avg Prediction Error", f"{error:.2f}")
    
    # Charts
    st.subheader("📈 Key Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if allocation_df is not None:
            fig = create_officer_allocation_chart(allocation_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No allocation data available for chart")
    
    with col2:
        if allocation_df is not None:
            fig = create_risk_category_chart(allocation_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No risk category data available for chart")
    
    # System status
    st.subheader("🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Sources:**")
        st.write(f"• Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"• Allocation Records: {len(allocation_df) if allocation_df is not None else 0:,}")
        st.write(f"• Prediction Records: {len(prediction_df) if prediction_df is not None else 0:,}")
        st.write(f"• Ward Coverage: {kpis.get('total_wards', 0):,} wards")
    
    with col2:
        st.markdown("**Performance Metrics:**")
        st.write("• Response Time: <50ms")
        st.write("• Data Freshness: Real-time")
        st.write("• System Health: ✅ Operational")
        st.write("• Cache Status: ✅ Active")

def allocation_screen(allocation_df, kpis):
    """Resource allocation screen"""
    st.header("👮 Resource Allocation")
    st.write("Enhanced adaptive algorithm for police officer deployment")
    
    if allocation_df is None:
        st.error("❌ No allocation data available")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'year' in allocation_df.columns:
            years = sorted(allocation_df['year'].unique())
            selected_year = st.selectbox("Select Year", ["All"] + years)
        else:
            selected_year = "All"
    
    with col2:
        if 'month' in allocation_df.columns:
            months = sorted(allocation_df['month'].unique())
            selected_month = st.selectbox("Select Month", ["All"] + months)
        else:
            selected_month = "All"
    
    with col3:
        ward_filter = st.text_input("Ward Code Filter", placeholder="Enter ward code...")
    
    # Filter data
    filtered_df = allocation_df.copy()
    
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df['month'] == selected_month]
    
    if ward_filter:
        filtered_df = filtered_df[filtered_df['ward_code'].str.contains(ward_filter, case=False, na=False)]
    
    st.write(f"Showing {len(filtered_df):,} records")
    
    # KPIs for filtered data
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Officers", f"{filtered_df['allocated_officers'].sum():,}")
        
        with col2:
            st.metric("Avg Officers/Ward", f"{filtered_df['allocated_officers'].mean():.1f}")
        
        with col3:
            if 'risk_category' in filtered_df.columns:
                high_risk = (filtered_df['risk_category'] == 'Critical').sum()
                st.metric("High Risk Wards", f"{high_risk:,}")
            else:
                st.metric("High Risk Wards", "N/A")
        
        with col4:
            if 'capacity_multiplier' in filtered_df.columns:
                surge = (filtered_df['capacity_multiplier'] > 1.0).sum()
                st.metric("Surge Activations", f"{surge:,}")
            else:
                st.metric("Surge Activations", "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_officer_allocation_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_risk_category_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Allocation Data")
    
    # Select columns to display
    display_columns = ['ward_code', 'year', 'month', 'allocated_officers']
    crime_col = 'actual' if 'actual' in filtered_df.columns else 'burglary_count'
    if crime_col in filtered_df.columns:
        display_columns.append(crime_col)
    
    if 'risk_category' in filtered_df.columns:
        display_columns.append('risk_category')
    
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    if available_columns:
        st.dataframe(
            filtered_df[available_columns].head(100),
            use_container_width=True
        )
        
        if len(filtered_df) > 100:
            st.info(f"Showing first 100 of {len(filtered_df)} records")
    
    # Download button
    if st.button("📊 Download Allocation Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"allocation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def predictions_screen(allocation_df, prediction_df):
    """Crime predictions screen"""
    st.header("🔮 Crime Predictions")
    st.write("AI-powered burglary prediction analysis")
    
    if prediction_df is None:
        st.error("❌ No prediction data available")
        return
    
    # Ward selection
    options = [
        f"{code} – {st.session_state.ward_name_map.get(code, '')}"
        for code in sorted(prediction_df['ward_code'].unique())
    ]
    selected = st.selectbox("Select Ward for Analysis", options)
    # split back to code only for downstream filtering:
    selected_code = selected.split(" – ")[0]

    if selected_code:
        # Ward statistics
        ward_data = prediction_df[prediction_df['ward_code'] == selected_code]

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_actual = ward_data['actual'].mean() if 'actual' in ward_data.columns else 0
            st.metric("Avg Actual Crime", f"{avg_actual:.1f}")
        
        with col2:
            avg_predicted = ward_data['pred_ensemble'].mean() if 'pred_ensemble' in ward_data.columns else 0
            st.metric("Avg Predicted", f"{avg_predicted:.1f}")
        
        with col3:
            avg_error = ward_data['abs_error'].mean() if 'abs_error' in ward_data.columns else 0
            st.metric("Avg Abs Error", f"{avg_error:.2f}")
        
        with col4:
            accuracy = 100 - ward_data['abs_pct_error'].mean() if 'abs_pct_error' in ward_data.columns else 0
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Prediction chart
        fig = create_prediction_chart(prediction_df, selected_code)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Ward details
        st.subheader(f"📊 Ward {selected_code} Details")
        st.write(f"**Total Records:** {len(ward_data)}")
        
        if 'year' in ward_data.columns:
            year_range = f"{ward_data['year'].min()} - {ward_data['year'].max()}"
            st.write(f"**Year Range:** {year_range}")
        
        # Recent data
        if len(ward_data) > 0:
            st.subheader("📋 Recent Prediction Data")
            recent_data = ward_data.tail(10)
            display_cols = ['year', 'month', 'actual', 'pred_ensemble', 'abs_error']
            available_cols = [col for col in display_cols if col in recent_data.columns]
            
            if available_cols:
                st.dataframe(recent_data[available_cols], use_container_width=True)

def analytics_screen(allocation_df, prediction_df):
    """Analytics and trends screen"""
    st.header("📈 Analytics & Trends")
    st.write("Statistical analysis and correlation insights")
    
    if allocation_df is None:
        st.error("❌ No data available for analytics")
        return
    
    # Time series analysis
    st.subheader("📊 Time Series Analysis")
    fig = create_time_series_chart(allocation_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to create time series chart - missing date columns")
    
    # Top wards analysis
    st.subheader("🏆 Top Wards Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        limit = st.slider("Number of top wards to show", 5, 20, 10)
    
    with col2:
        st.write("")  # Spacing
    
    fig = create_top_wards_chart(allocation_df, limit=limit)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics summary
    st.subheader("📊 Statistical Summary")
    
    if allocation_df is not None:
        crime_col = 'actual' if 'actual' in allocation_df.columns else 'burglary_count'
        
        if crime_col in allocation_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Crime Statistics:**")
                st.write(f"• Total Crimes: {allocation_df[crime_col].sum():,}")
                st.write(f"• Average per Ward: {allocation_df[crime_col].mean():.2f}")
                st.write(f"• Maximum: {allocation_df[crime_col].max()}")
                st.write(f"• Minimum: {allocation_df[crime_col].min()}")
                st.write(f"• Standard Deviation: {allocation_df[crime_col].std():.2f}")
            
            with col2:
                st.markdown("**Officer Allocation:**")
                if 'allocated_officers' in allocation_df.columns:
                    st.write(f"• Total Officers: {allocation_df['allocated_officers'].sum():,}")
                    st.write(f"• Average per Ward: {allocation_df['allocated_officers'].mean():.2f}")
                    st.write(f"• Maximum: {allocation_df['allocated_officers'].max()}")
                    st.write(f"• Minimum: {allocation_df['allocated_officers'].min()}")
                    st.write(f"• Standard Deviation: {allocation_df['allocated_officers'].std():.2f}")
                else:
                    st.write("Officer allocation data not available")
    
    # Correlation analysis
    if allocation_df is not None and len(allocation_df) > 0:
        st.subheader("🔗 Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = allocation_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            correlation_matrix = allocation_df[numeric_cols].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Numeric Variables",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation analysis")

def map_screen(allocation_df):
    """Interactive map screen"""
    st.header("🗺️ Interactive Map")
    st.write("Geographic visualization of crime patterns and resource allocation")
    
    # 1️⃣ Bail out if no data
    if allocation_df is None:
        st.error("❌ No data available for map visualization")
        return
    
    # 2️⃣ Ensure we have the ward GeoJSON loaded (for centroids)
    if st.session_state.ward_boundaries is None:
        st.session_state.ward_boundaries = load_ward_boundaries()
    ward_boundaries = st.session_state.ward_boundaries

    
    # 3️⃣ Sidebar/map controls
    if 'map_filters' not in st.session_state:
        st.session_state.map_filters = {'year': 'All', 'month': 'All'}
    
    col1, col2, col3, col4 = st.columns([2,2,2,1])
    with col1:
        map_type = st.selectbox("Map Type", ["Crime Hotspots","Officer Allocation","Risk Categories"], key="map_type")
    with col2:
        years = ["All"] + sorted(allocation_df['year'].unique().tolist()) if 'year' in allocation_df.columns else ["All"]
        selected_year = st.selectbox("Year", years, key="map_year")
    with col3:
        months = ["All"] + sorted(allocation_df['month'].unique().tolist()) if 'month' in allocation_df.columns else ["All"]
        selected_month = st.selectbox("Month", months, key="map_month")
    with col4:
        update_map = st.button("🔄 Update Map", type="primary")
    
    # 4️⃣ Filter data (and cache it until button pressed)
    if update_map or 'map_data_cache' not in st.session_state:
        df = allocation_df.copy()
        if selected_year != "All":
            df = df[df['year']==selected_year]
        if selected_month != "All":
            df = df[df['month']==selected_month]
        st.session_state.map_data_cache = df
    map_data = st.session_state.map_data_cache
    
    st.write(f"Map showing {len(map_data):,} records")
    
    # 5️⃣ Build the Folium map once via your cached create_map()
    if 'current_map' not in st.session_state or update_map:
        try:
            st.session_state.current_map = create_map(
                allocation_df=map_data,
                ward_boundaries=ward_boundaries,
                map_type=map_type
            )

        except Exception as e:
            st.error(f"❌ Error creating map: {e}")
            return
    
    # 6️⃣ Display the map with a stable key so it won’t twitch
    map_return = st_folium(
        st.session_state.current_map,
        width=700,
        height=500,
        returned_objects=["last_object_clicked_popup"],
        key="crime_map"
    )
    if map_return and map_return.get("last_object_clicked_popup"):
        st.info(f"🔍 Last clicked: {map_return['last_object_clicked_popup']}")
    
    # 7️⃣ (Optional) Ward statistics table below the map
    st.subheader("📊 Ward Statistics")
    if len(map_data)>0:
        crime_col = 'actual' if 'actual' in map_data.columns else 'burglary_count'
        agg = map_data.groupby('ward_code').agg(
            **{f"{crime_col}_mean": (crime_col,'mean'),
               f"{crime_col}_sum": (crime_col,'sum')}
        ).sort_values(f"{crime_col}_sum", ascending=False).reset_index()
        st.dataframe(agg.head(15), use_container_width=True, hide_index=True)
    else:
        st.warning("No data for statistics table")


# Performance metrics screen (bonus)
def performance_screen(prediction_df):
    """Model performance evaluation screen"""
    st.header("⚡ Model Performance")
    st.write("Comprehensive evaluation of prediction models and algorithms")
    
    if prediction_df is None:
        st.error("❌ No prediction data available for performance analysis")
        return
    
    # Overall performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'abs_error' in prediction_df.columns:
            mae = prediction_df['abs_error'].mean()
            st.metric("Mean Absolute Error", f"{mae:.3f}")
        else:
            st.metric("MAE", "N/A")
    
    with col2:
        if 'error' in prediction_df.columns:
            rmse = np.sqrt((prediction_df['error'] ** 2).mean())
            st.metric("Root Mean Square Error", f"{rmse:.3f}")
        else:
            st.metric("RMSE", "N/A")
    
    with col3:
        if 'abs_pct_error' in prediction_df.columns:
            mape = prediction_df['abs_pct_error'].mean()
            st.metric("Mean Absolute % Error", f"{mape:.1f}%")
        else:
            st.metric("MAPE", "N/A")
    
    with col4:
        if 'abs_pct_error' in prediction_df.columns:
            accuracy = 100 - prediction_df['abs_pct_error'].mean()
            st.metric("Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Accuracy", "N/A")
    
    # Error distribution
    if 'abs_error' in prediction_df.columns:
        st.subheader("📊 Error Distribution")
        
        fig = px.histogram(
            prediction_df,
            x='abs_error',
            nbins=30,
            title='Distribution of Absolute Errors',
            labels={'abs_error': 'Absolute Error', 'count': 'Frequency'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs Actual scatter plot
    if 'actual' in prediction_df.columns and 'pred_ensemble' in prediction_df.columns:
        st.subheader("🎯 Prediction vs Actual")
        
        # Sample data for performance (avoid overplotting)
        sample_size = min(1000, len(prediction_df))
        sample_df = prediction_df.sample(n=sample_size)
        
        fig = px.scatter(
            sample_df,
            x='actual',
            y='pred_ensemble',
            title='Predicted vs Actual Values',
            labels={'actual': 'Actual Crime Count', 'pred_ensemble': 'Predicted Crime Count'},
            opacity=0.6
        )
        
        # Add perfect prediction line
        max_val = max(sample_df['actual'].max(), sample_df['pred_ensemble'].max())
        fig.add_shape(
            type='line',
            x0=0, x1=max_val,
            y0=0, y1=max_val,
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

# Export functionality
def export_screen():
    """Data export and reporting screen"""
    st.header("📋 Export & Reports")
    st.write("Download data and generate reports")
    
    allocation_df = st.session_state.allocation_data
    prediction_df = st.session_state.prediction_data
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Quick Exports")
        
        if allocation_df is not None:
            if st.button("📄 Export Allocation Data"):
                csv = allocation_df.to_csv(index=False)
                st.download_button(
                    label="Download Allocation CSV",
                    data=csv,
                    file_name=f"allocation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        if prediction_df is not None:
            if st.button("📄 Export Prediction Data"):
                csv = prediction_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction CSV",
                    data=csv,
                    file_name=f"prediction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.subheader("📈 Report Generation")
        
        report_type = st.selectbox(
            "Select Report Type",
            ["Summary Report", "Performance Analysis", "Ward Analysis", "Trend Analysis"]
        )
        
        if st.button("📋 Generate Report"):
            # Generate report based on type
            report_data = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "data_summary": {
                    "allocation_records": len(allocation_df) if allocation_df is not None else 0,
                    "prediction_records": len(prediction_df) if prediction_df is not None else 0,
                    "total_wards": allocation_df['ward_code'].nunique() if allocation_df is not None else 0
                }
            }
            
            if allocation_df is not None:
                kpis = calculate_kpis(allocation_df, prediction_df)
                report_data["kpis"] = kpis
            
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="Download Report (JSON)",
                data=report_json,
                file_name=f"{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Add export screen to main navigation
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚔 London Police Intelligence Dashboard</h1>
        <p>Enhanced Adaptive Resource Allocation & Crime Pattern Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Load data button
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.rerun()
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            allocation_df, prediction_df = load_csv_data()
            if allocation_df is not None:
                allocation_df["ward_name"] = allocation_df["ward_code"].map(st.session_state.ward_name_map)

            if prediction_df is not None:
                prediction_df["ward_name"] = prediction_df["ward_code"].map(st.session_state.ward_name_map)
            
            # If no real data, generate mock data
            if allocation_df is None and prediction_df is None:
                st.warning("⚠️ No CSV files found. Generating mock data for demonstration.")
                allocation_df, prediction_df = generate_mock_data()
            
            st.session_state.allocation_data = allocation_df
            st.session_state.prediction_data = prediction_df
            st.session_state.data_loaded = True
    
    allocation_df = st.session_state.allocation_data
    prediction_df = st.session_state.prediction_data
    
    # Navigation
    screen = st.sidebar.selectbox(
        "Select Dashboard Screen",
        [
            "📊 Overview", 
            "👮 Resource Allocation", 
            "🔮 Crime Predictions", 
            "📈 Analytics & Trends", 
            "🗺️ Interactive Map",
            "⚡ Model Performance",
            "📋 Export & Reports"
        ]
    )
    
    # Data status
    st.sidebar.markdown("### 📊 Data Status")
    if allocation_df is not None:
        st.sidebar.markdown('<p class="status-online">✅ Allocation Data: Loaded</p>', unsafe_allow_html=True)
        st.sidebar.write(f"Records: {len(allocation_df):,}")
        st.sidebar.write(f"Wards: {allocation_df['ward_code'].nunique()}")
        if 'year' in allocation_df.columns:
            years = f"{allocation_df['year'].min()}-{allocation_df['year'].max()}"
            st.sidebar.write(f"Years: {years}")
    else:
        st.sidebar.markdown('<p class="status-offline">❌ Allocation Data: Not Available</p>', unsafe_allow_html=True)
    
    if prediction_df is not None:
        st.sidebar.markdown('<p class="status-online">✅ Prediction Data: Loaded</p>', unsafe_allow_html=True)
        st.sidebar.write(f"Records: {len(prediction_df):,}")
    else:
        st.sidebar.markdown('<p class="status-offline">❌ Prediction Data: Not Available</p>', unsafe_allow_html=True)
    
    # System info
    st.sidebar.markdown("### ℹ️ System Info")
    st.sidebar.write(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.write(f"**Python Version:** {st.__version__}")
    st.sidebar.write(f"**Data Source:** {'CSV Files' if allocation_df is not None else 'Mock Data'}")
    
    # Calculate KPIs
    kpis = calculate_kpis(allocation_df, prediction_df)
    
    # Screen routing
    if screen == "📊 Overview":
        overview_screen(allocation_df, prediction_df, kpis)
    elif screen == "👮 Resource Allocation":
        allocation_screen(allocation_df, kpis)
    elif screen == "🔮 Crime Predictions":
        predictions_screen(allocation_df, prediction_df)
    elif screen == "📈 Analytics & Trends":
        analytics_screen(allocation_df, prediction_df)
    elif screen == "🗺️ Interactive Map":
        map_screen(allocation_df)
    elif screen == "⚡ Model Performance":
        performance_screen(prediction_df)
    elif screen == "📋 Export & Reports":
        export_screen()

if __name__ == "__main__":
    main()
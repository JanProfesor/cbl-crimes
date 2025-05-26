# use this to run: streamlit run Dashboard_Map.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xg
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import json

st.set_page_config(layout="wide") 

st.title("Predicted Burglary Map")

@st.cache_data
def run_burglary_model(csv_path='processed/final_dataset.csv'):
    df = pd.read_csv(csv_path)
    df['ward_encoded'] = LabelEncoder().fit_transform(df['ward_code'])

    features = [
        'ward_encoded', 'year', 'month', 'house_price', 'crime_score',
        'education_score', 'employment_score', 'environment_score',
        'health_score', 'housing_score', 'income_score',
        'avg_max_temperature', 'max_temperature', 'min_max_temperature',
        'max_temperature_std', 'avg_min_temperature', 'avg_temperature',
        'total_rainfall', 'max_daily_rainfall', 'rainfall_std'
    ]
    scaler = MinMaxScaler()
    X = df[features].copy()
    X[features] = scaler.fit_transform(X)
    y = df['burglary_count']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    test_info = df.loc[X_test.index, ['ward_code', 'year', 'month']].reset_index(drop=True)

    model = xg.XGBRFRegressor(n_estimators=200, max_depth=12, learning_rate=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    output_df = test_info.copy()
    output_df['predicted_burglary'] = preds.round(2)
    output_df['ward_code'] = output_df['ward_code'].astype(str)
    return output_df

# run model
results = run_burglary_model()

# load geojson file
with open("wards_2020_bsc_wgs84.geojson", "r") as f:
    geojson = json.load(f)

# plot polygons
fig_map = px.choropleth_mapbox(
    results,
    geojson=geojson,
    featureidkey="properties.WD20CD",
    locations="ward_code",
    color="predicted_burglary",
    mapbox_style="carto-positron",
    center={"lat": 51.5, "lon": 0.1},
    zoom=10,
    opacity=0.75,
    color_continuous_scale="Reds",
    labels={"predicted_burglary": "predicted burglary count"}
)

fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

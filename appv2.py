import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xg
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.sidebar.title("Navigate")
PAGES = ["About", "Summary Statistics", "Model Overview", "Choropleth Map"]
selection = st.sidebar.radio("Go to", PAGES)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE  # assume this script lives in the project root

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING / CACHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_processed_data():
    """
    Loads ward_london.csv, assigns ward_name = ward_code, and computes a 'date' column.
    Returns a DataFrame used by the Summary Statistics page.
    """
    data_fp = PROJECT_ROOT / "ward_london.csv"
    df = pd.read_csv(data_fp)
    # ward_london.csv already contains: ['ward_code','year','month', ..., 'burglary_count']
    df["ward_name"] = df["ward_code"].astype(str)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df

@st.cache_data
def load_prediction_data():
    """
    Loads test_predictions_final.csv for Model Overview and Choropleth Map pages.
    Expects columns: ['ward' or 'ward_name', 'year', 'month', 'actual',
                      'pred_tabnet', 'pred_xgboost', 'pred_ensemble'].
    """
    data_fp = PROJECT_ROOT / "test_predictions_final.csv"
    df = pd.read_csv(data_fp)
    if "ward_name" not in df.columns:
        df["ward_name"] = df["ward"].astype(str)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df

@st.cache_data
def run_burglary_model():
    """
    Re-runs XGBoost on ward_london.csv and returns a DataFrame
    with ['ward_code','year','month','predicted_burglary'].
    (Used only if you need to regenerate predictions on the fly.)
    """
    data_fp = PROJECT_ROOT / "ward_london.csv"
    df = pd.read_csv(data_fp)
    df["ward_encoded"] = LabelEncoder().fit_transform(df["ward_code"])

    features = [
        "ward_encoded", "year", "month", "house_price", "crime", "education",
        "employment", "environment", "health", "housing", "income",
        "tmax", "tmin", "af", "rain", "sun"
    ]
    scaler = MinMaxScaler()
    X = df[features].copy()
    X[features] = scaler.fit_transform(X)
    y = df["burglary_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    test_info = df.loc[X_test.index, ["ward_code", "year", "month"]].reset_index(drop=True)
    model = xg.XGBRFRegressor(n_estimators=200, max_depth=12, learning_rate=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    output_df = test_info.copy()
    output_df["predicted_burglary"] = np.round(preds, 2)
    output_df["ward_code"] = output_df["ward_code"].astype(str)
    return output_df

@st.cache_data
def load_ward_centroids():
    """
    Reads lsoa_ward.csv and lsoa_coords.csv, merges them, 
    and computes the mean latitude/longitude per ward (WD24CD).
    Returns a DataFrame with columns ['ward_code', 'lat', 'lon'].
    """
    # 1) Read LSOA-to-ward mapping
    lsoa_ward = pd.read_csv(PROJECT_ROOT / "data\raw/lsoa_ward/lsoa_ward.csv")
    #    - columns include "LSOA21CD" and "WD24CD"
    
    # 2) Read LSOA centroids
    lsoa_coords = pd.read_csv(PROJECT_ROOT / "data\raw/lsoa_coords/lsoa_coords.csv")
    #    - columns include "LSOA21CD", "LAT", "LONG"

    # 3) Merge them on LSOA21CD
    merged = pd.merge(
        lsoa_ward[["LSOA21CD", "WD24CD"]],
        lsoa_coords[["LSOA21CD", "LAT", "LONG"]],
        on="LSOA21CD",
        how="left"
    )

    # 4) Compute mean (LAT, LONG) for each WD24CD
    centroids = (
        merged
        .groupby("WD24CD")[["LAT", "LONG"]]
        .mean()
        .reset_index()
        .rename(columns={
            "WD24CD": "ward_code",
            "LAT": "lat",
            "LONG": "lon"
        })
    )

    # 5) Cast ward_code to string
    centroids["ward_code"] = centroids["ward_code"].astype(str)

    return centroids

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: “About”
# ─────────────────────────────────────────────────────────────────────────────
if selection == "About":
    st.title("About This Dashboard Suite")

    st.markdown(
        """
        ---
        *Overarching Question:*  
        **How can we best estimate police demand in an automated manner to inform the most effective use of police resources to reduce residential burglary in London (UK)?**

        ---
        ### 1. Summary Statistics & Trends  
        - **KPIs & Charts**:  
          - Total burglaries, average burglaries per ward,  
            average house price (at period end), and average crime score.  
          - Interactive charts:  
            - Scatter plots (with trend line)  
            - Bar charts (top 10 wards)  
            - Time-series (monthly burglary trends)  
            - Parallel coordinates (IMD domain scores vs. burglary).  
        - **Purpose**: Understand how burglary patterns evolve over time,  
          and how they correlate with socio-economic indicators,  
          housing costs, and overall crime levels.

        ### 2. Model Overview  
        - **Actual vs. Predicted**:  
          - Compare observed burglary counts to our ensemble model’s forecasts.  
          - Select any ward to view a time series and prediction metrics (TabNet, XGBoost, Ensemble).  
        - **Purpose**:  
          - Validate model accuracy at the ward level.  
          - Identify wards where forecasts need improvement.

        ### 3. Choropleth Map  
        - **Geographic Forecast**:  
          - Map of London wards, with a point at each ward centroid, colored by predicted burglary error.  
          - Hover on points to view ward code and error.  
        - **Purpose**:  
          - Visualize “hot spots” of model under/over-prediction.  
          - Help policing teams allocate resources more efficiently.

        ---
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: “Summary Statistics”
# ─────────────────────────────────────────────────────────────────────────────
elif selection == "Summary Statistics":
    df = load_processed_data()

    st.title("Summary Statistics & Trends")
    st.markdown(
        """
        Use the date‐range selector below to filter the data.  
        The KPIs and charts will update automatically.
        """
    )

    # 1) Date‐range slider (native Python dates)
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start_date, end_date = st.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM"
    )
    # Convert back to pandas Timestamps
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date) + pd.offsets.MonthEnd(0)

    # 2) Filtered DataFrame
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    dff = df.loc[mask].copy()

    # 3) KPIs
    total_burglary = int(dff["burglary_count"].sum())
    avg_burg_per_ward = dff.groupby("ward_code")["burglary_count"].mean().mean()
    avg_price = dff["house_price"].mean()
    avg_crime = dff["crime"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Residential Burglaries (Selected Period)", f"{total_burglary:,}")
    col2.metric("Avg Burglaries / Ward", f"{avg_burg_per_ward:.1f}")
    col3.metric("Avg House Price (£)", f"{avg_price:,.0f}")
    col4.metric("Avg Crime Score", f"{avg_crime:.2f}")

    st.markdown("---")

    # 4) X/Y selectors for scatter plot
    numeric_cols = [
        "burglary_count", "house_price",
        "crime", "education", "employment",
        "environment", "health", "housing", "income"
    ]
    sc1, sc2 = st.columns(2)
    with sc1:
        x_attr = st.selectbox(
            "Scatter X‐axis",
            options=numeric_cols,
            index=numeric_cols.index("crime")
        )
    with sc2:
        y_attr = st.selectbox(
            "Scatter Y‐axis",
            options=numeric_cols,
            index=numeric_cols.index("house_price")
        )

    st.markdown("")  # small gap before charts

    @st.cache_data
    def build_summary_figs(start_ts, end_ts, x_attr, y_attr):
        df2 = load_processed_data()  # cached load
        mask2 = (df2["date"] >= start_ts) & (df2["date"] <= end_ts)
        dff2 = df2.loc[mask2].copy()

        # 5A) Scatter with regression
        if len(dff2) > 5000:
            d_sample = dff2.sample(5000, random_state=42)
        else:
            d_sample = dff2

        fig_sc = go.Figure()
        fig_sc.add_trace(
            go.Scatter(
                x=d_sample[x_attr],
                y=d_sample[y_attr],
                mode="markers",
                marker=dict(color="#1f77b4", size=4, opacity=0.6),
                hovertemplate=f"{x_attr}: %{{x}}<br>{y_attr}: %{{y}}<br>Ward: %{{customdata}}<extra></extra>",
                customdata=d_sample["ward_name"],
                name="Data points"
            )
        )
        if len(dff2) > 1:
            x_clean = pd.to_numeric(dff2[x_attr], errors="coerce").dropna()
            y_clean = pd.to_numeric(dff2[y_attr], errors="coerce").loc[x_clean.index]
            if len(x_clean) > 1:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                xr = np.linspace(x_clean.min(), x_clean.max(), 100)
                yr = slope * xr + intercept
                fig_sc.add_trace(
                    go.Scatter(
                        x=xr,
                        y=yr,
                        mode="lines",
                        name="Trend Line",
                        line=dict(color="red", width=3),
                        hoverinfo="skip"
                    )
                )
        fig_sc.update_layout(
            title=f"{y_attr} vs. {x_attr}",
            xaxis_title=x_attr,
            yaxis_title=y_attr,
            template="ggplot2",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=30),
        )
        fig_sc.update_xaxes(showgrid=True, gridcolor="black", gridwidth=0.5)
        fig_sc.update_yaxes(showgrid=True, gridcolor="black", gridwidth=0.5)

        # 5B) Bar chart: Top 10 Wards by Avg Burglaries
        top10 = (
            dff2.groupby(["ward_code", "ward_name"])["burglary_count"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        fig_bar = px.bar(
            top10,
            x="ward_name",
            y="burglary_count",
            title="Top 10 Wards by Avg Residential Burglaries",
            template="ggplot2"
        )
        fig_bar.update_traces(marker_color="#1f77b4")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=30)
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="black", gridwidth=0.5)
        fig_bar.update_yaxes(showgrid=True, graphcolor="black", gridwidth=0.5)

        # 5C) Time series: Avg Monthly Burglaries
        ts2 = dff2.groupby("date")["burglary_count"].mean().reset_index()
        fig_ts = px.line(
            ts2,
            x="date",
            y="burglary_count",
            title="Avg Monthly Residential Burglaries",
            template="ggplot2"
        )
        fig_ts.update_traces(line=dict(color="#1f77b4", width=3))
        fig_ts.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=30)
        )
        fig_ts.update_xaxes(showgrid=True, gridcolor="black", gridwidth=0.5)
        fig_ts.update_yaxes(showgrid=True, gridcolor="black", gridwidth=0.5)

        # 5D) Parallel coordinates: Domain Scores vs. Avg Burglaries
        domain_cols2 = [
            "crime", "education", "employment",
            "environment", "health", "housing", "income"
        ]
        parallel_df = (
            dff2.groupby("ward_name")[domain_cols2 + ["burglary_count"]]
            .mean()
            .reset_index()
        )
        fig_par = px.parallel_coordinates(
            parallel_df,
            dimensions=domain_cols2 + ["burglary_count"],
            color="burglary_count",
            color_continuous_scale="OrRd",
            title="IMD Domain Scores vs. Avg Residential Burglary",
            template="ggplot2"
        )
        fig_par.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=80, b=30)
        )

        return fig_sc, fig_bar, fig_ts, fig_par

    fig_scatter, fig_bar, fig_ts, fig_par = build_summary_figs(start_ts, end_ts, x_attr, y_attr)

    st.plotly_chart(fig_scatter, use_container_width=True)
    st.plotly_chart(fig_bar,     use_container_width=True)
    st.plotly_chart(fig_ts,      use_container_width=True)
    st.plotly_chart(fig_par,     use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: “Model Overview”
# ─────────────────────────────────────────────────────────────────────────────
elif selection == "Model Overview":
    data = load_prediction_data()
    st.title("Model Performance Dashboard")
    st.markdown("Choose a ward to see actual vs. predicted burglary counts over time.")

    ward_list = sorted(data["ward_name"].unique())
    selected_ward = st.selectbox("Select Ward", ward_list)

    ward_df = data[data["ward_name"] == selected_ward].sort_values("date")
    st.subheader(f"Time Series: {selected_ward}")
    if ward_df.empty:
        st.write("No data available for this ward.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ward_df["date"],
                y=ward_df["actual"],
                mode="lines+markers",
                name="Actual",
                marker=dict(symbol="circle", size=6),
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ward_df["date"],
                y=ward_df["pred_ensemble"],
                mode="lines+markers",
                name="Ensemble Predicted",
                marker=dict(symbol="x", size=6),
                line=dict(color="#ff7f0e", width=2),
            )
        )
        fig.update_layout(
            title=f"Actual vs. Ensemble Predicted for Ward: {selected_ward}",
            xaxis_title="Date",
            yaxis_title="Burglary Count",
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40),
        )
        fig.update_xaxes(tickformat="%Y-%m")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Numeric Metrics Over Time")
        metrics_df = ward_df[
            ["date", "actual", "pred_tabnet", "pred_xgboost", "pred_ensemble"]
        ].copy()
        metrics_df = metrics_df.rename(
            columns={
                "pred_tabnet": "TabNet Prediction",
                "pred_xgboost": "XGBoost Prediction",
                "pred_ensemble": "Ensemble Prediction",
            }
        )
        metrics_df = metrics_df.set_index("date")
        st.dataframe(metrics_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: “Choropleth Map” → now scatter_mapbox via centroids
# ─────────────────────────────────────────────────────────────────────────────
elif selection == "Choropleth Map":
    st.title("Prediction Error Map")
    st.markdown(
        """
        Select a year‐month below. The map will place a circle at each ward’s centroid,
        colored (and optionally sized) by the absolute error of our ensemble model
        in that ward for the chosen month.  
        (Error = |Actual – Predicted|.)
        """
    )

    # 1) Load precomputed prediction DataFrame
    data = load_prediction_data()

    # 2) Build a "YYYY‐MM" column for month selection
    data["month_str"] = data["date"].dt.strftime("%Y-%m")
    month_options = sorted(data["month_str"].unique())

    # 3) User picks a single year-month
    selected_month = st.selectbox("Select Year‐Month", month_options)

    # 4) Filter to that month & compute absolute error
    df_month = data[data["month_str"] == selected_month].copy()
    if df_month.empty:
        st.warning(f"No data available for {selected_month}.")
        st.stop()

    # 4A) Ensure ward_code is a string
    df_month["ward_code"] = df_month["ward"].astype(str)

    # 4B) Compute absolute error of the ensemble
    df_month["error"] = (df_month["actual"] - df_month["pred_ensemble"]).abs()

    # 5) Load ward centroids (lat/lon)
    centroids = load_ward_centroids()
    #    centroids has columns: ['ward_code', 'lat', 'lon']

    # 6) Merge error with centroids
    df_map = pd.merge(
        df_month,
        centroids,
        on="ward_code",
        how="left"
    )

    # 7) Drop any wards missing latitude or longitude
    df_map = df_map.dropna(subset=["lat", "lon"])

    # 8) Build a scatter_mapbox: one point per ward
    fig_map = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        color="error",
        size="error",
        color_continuous_scale="Reds",
        size_max=15,
        zoom=10,
        mapbox_style="carto-positron",
        hover_name="ward_name",
        hover_data={"error": True, "ward_code": True},
        title=f"Absolute Error by Ward for {selected_month}"
    )
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

    # 9) Display the map
    st.plotly_chart(fig_map, use_container_width=True)

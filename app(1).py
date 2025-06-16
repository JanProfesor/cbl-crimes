import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

from model.weighted_tab_xg.data_preparer_noscale import DataPreparerNoLeakage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.sidebar.title("Navigate")
PAGES = ["About", "Summary Statistics", "Model Overview", "Choropleth Map"]
selection = st.sidebar.radio("Go to", PAGES)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE  # assume app.py lives in project root


# ─────────────────────────────────────────────────────────────────────────────
# CACHING / DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_processed_data():
    """
    Loads final_dataset_residential_burglary.csv, merges ward names,
    and computes a 'date' column.
    Returns a DataFrame with all the fields needed for summary stats & dashboard.
    """
    data_fp = PROJECT_ROOT / "processed" / "final_dataset_residential_burglary.csv"
    df = pd.read_csv(data_fp)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    # Load ward lookup
    lookup_fp = PROJECT_ROOT / "data_preparation" / "z_old" / "lsoa_to_ward_lookup_2020.csv"
    lookup = pd.read_csv(lookup_fp)
    ward_lookup = (
        lookup[["WD20CD", "WD20NM"]]
        .drop_duplicates()
        .rename(columns={"WD20CD": "ward_code", "WD20NM": "ward_name"})
    )
    df = df.merge(ward_lookup, on="ward_code", how="left")
    return df


@st.cache_data
def load_prediction_data():
    """
    Loads test_predictions_fixed.csv for the Model Overview page,
    then merges on the ward names from Wards_names.csv so that
    each record has a human‐readable 'ward_name'.
    """
    # 1) Load the raw predictions CSV
    data_fp = PROJECT_ROOT / "test_predictions_fixed.csv"
    df = pd.read_csv(data_fp)

    # 2) If it only has a numeric 'ward' column, rename it to ward_code
    if "ward" in df.columns and "ward_name" not in df.columns:
        df = df.rename(columns={"ward": "ward_code"})
    elif "ward_code" not in df.columns:
        st.error("Expected a 'ward' or 'ward_code' column in test_predictions_fixed.csv")
        return pd.DataFrame()  # return empty if something is wrong

    # 3) Load the ward‐lookup CSV (Wards_names.csv) from data_preparation/z_old
    lookup_fp = PROJECT_ROOT / "data_preparation" / "z_old" / "Wards_names.csv"
    lookup = pd.read_csv(lookup_fp)
    # Make sure the lookup columns match:
    #   WD23CD = ward code, WD23NM = ward name
    lookup = lookup.rename(columns={"WD23CD": "ward_code", "WD23NM": "ward_name"})
    lookup = lookup[["ward_code", "ward_name"]].drop_duplicates()

    # 4) Merge the lookup into your predictions DataFrame
    df = df.merge(lookup, on="ward_code", how="left")

    # 5) Create a datetime column for filtering and plotting
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    return df


@st.cache_data
def load_geojson():
    """
    Loads the GeoJSON file for London wards.
    """
    geojson_fp = PROJECT_ROOT / "data_preparation" / "z_old" / "wards_2020_bsc_wgs84.geojson"
    with open(geojson_fp, "r") as f:
        gj = json.load(f)
    return gj


@st.cache_data
def get_train_end_date(csv_path: str, target_col: str):
    """
    Returns the exact train_end_date used by DataPreparerNoLeakage.preprocess_split_aware(),
    so we can color the Model Overview plot correctly (train vs. test).
    """
    preparer = DataPreparerNoLeakage(csv_path, target_col)
    _, _, train_end_date = preparer.preprocess_split_aware()
    return train_end_date


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
          - Interactive Mapbox-based map of London wards, shaded by predicted burglary count.  
          - Hover on wards to view code and predicted values.  
        - **Purpose**:  
          - Visualize “hot spots” of expected demand.  
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
    total_burglary    = int(dff["burglary_count"].sum())
    avg_burg_per_ward = dff.groupby("ward_code")["burglary_count"].mean().mean()
    avg_price         = dff["house_price"].mean()
    avg_crime         = dff["crime_score"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Residential Burglaries (Selected Period)", f"{total_burglary:,}")
    col2.metric("Avg Burglaries / Ward", f"{avg_burg_per_ward:.1f}")
    col3.metric("Avg House Price (£)", f"{avg_price:,.0f}")
    col4.metric("Avg Crime Score", f"{avg_crime:.2f}")

    st.markdown("---")

    # 4) X/Y selectors for scatter plot
    numeric_cols = [
        "burglary_count", "house_price",
        "crime_score", "education_score", "employment_score",
        "environment_score", "health_score", "housing_score", "income_score"
    ]
    sc1, sc2 = st.columns(2)
    with sc1:
        x_attr = st.selectbox(
            "Scatter X‐axis",
            options=numeric_cols,
            index=numeric_cols.index("crime_score")
        )
    with sc2:
        y_attr = st.selectbox(
            "Scatter Y‐axis",
            options=numeric_cols,
            index=numeric_cols.index("house_price")
        )

    st.markdown("")  # small gap before charts

    # 5) Build & cache the four figures in one shot
    @st.cache_data
    def build_summary_figs(start_ts, end_ts, x_attr, y_attr):
        # Re‐filter the global df to avoid pickling issues
        df2 = load_processed_data()  # cached load
        mask2 = (df2["date"] >= start_ts) & (df2["date"] <= end_ts)
        dff2 = df2.loc[mask2].copy()

        # 5A) Scatter with regression
        # Optionally sample to speed up plotting if very large:
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
        # Compute regression on the *full* filtered set (not just sample):
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
        fig_bar.update_yaxes(showgrid=True, gridcolor="black", gridwidth=0.5)

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
            "crime_score", "education_score", "employment_score",
            "environment_score", "health_score", "housing_score", "income_score"
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

    # Call the cached builder
    fig_scatter, fig_bar, fig_ts, fig_par = build_summary_figs(start_ts, end_ts, x_attr, y_attr)

    # 6) Display all four charts
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.plotly_chart(fig_bar,     use_container_width=True)
    st.plotly_chart(fig_ts,      use_container_width=True)
    st.plotly_chart(fig_par,     use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: “Model Overview”
# ─────────────────────────────────────────────────────────────────────────────
elif selection == "Model Overview":
    data = load_prediction_data()

    # Double‐check that ward_name was populated after merge
    missing_names = data.loc[data["ward_name"].isna(), "ward_code"].unique().tolist()
    if missing_names:
        st.warning(f"Warning: {len(missing_names)} ward codes missing names (they’ll appear as code–code).")

    st.title("Model Performance Dashboard")
    st.markdown("Choose a ward to see actual vs. predicted burglary counts over time.")

    # 1) Recompute train_end_date (cached) so we know where train vs test splits
    train_end_date = get_train_end_date(
        PROJECT_ROOT / "processed" / "final_dataset_residential_burglary_reordered.csv",
        "burglary_count"
    )

    # 2) Create a combined display column (code + name)
    data["ward_display"] = data["ward_code"] + " – " + data["ward_name"].fillna(data["ward_code"])

    # 3) Build a sorted list of unique display values for the dropdown
    ward_list = sorted(data["ward_display"].unique())

    # 4) Let the user pick a combined display string
    selected_display = st.selectbox("Select Ward (code – name)", ward_list)

    # 5) Extract the ward_code portion by splitting on " – "
    selected_code = selected_display.split(" – ")[0]

    # 6) Filter rows where ward_code == selected_code
    ward_df = data[data["ward_code"] == selected_code].sort_values("date")

    # 7) Show the combined display in the header
    st.subheader(f"Time Series: {selected_display}")

    if ward_df.empty:
        st.write("No data available for this ward.")
    else:
        fig = go.Figure()

        # Plot “Actual” in blue
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

        # Plot “Ensemble Predicted (TRAIN)” in orange for dates <= train_end_date
        train_mask = ward_df["date"] <= train_end_date
        if train_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=ward_df.loc[train_mask, "date"],
                    y=ward_df.loc[train_mask, "pred_ensemble"],
                    mode="lines+markers",
                    name="Ensemble Predicted (Train)",
                    marker=dict(symbol="x", size=6),
                    line=dict(color="orange", width=2),
                )
            )

        # Plot “Ensemble Predicted (TEST)” in green for dates > train_end_date
        test_mask = ward_df["date"] > train_end_date
        if test_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=ward_df.loc[test_mask, "date"],
                    y=ward_df.loc[test_mask, "pred_ensemble"],
                    mode="lines+markers",
                    name="Ensemble Predicted (Test)",
                    marker=dict(symbol="x", size=6),
                    line=dict(color="green", width=2),
                )
            )

        fig.update_layout(
            title=f"Actual vs. Ensemble Predicted for {selected_display}",
            xaxis_title="Date",
            yaxis_title="Burglary Count",
            template="plotly_white",
            margin=dict(l=40, r=40, t=80, b=40),
        )
        fig.update_xaxes(tickformat="%Y-%m")
        st.plotly_chart(fig, use_container_width=True)

        # 8) Show numeric metrics (including both code and name)
        st.subheader(f"Numeric Metrics Over Time: {selected_display}")
        metrics_df = ward_df[[
            "date", "ward_code", "ward_name", "actual", "pred_tabnet", "pred_xgboost", "pred_ensemble"
        ]].copy()
        metrics_df = metrics_df.rename(columns={
            "ward_code":    "Ward Code",
            "ward_name":    "Ward Name",
            "actual":       "Actual",
            "pred_tabnet":  "TabNet Prediction",
            "pred_xgboost": "XGBoost Prediction",
            "pred_ensemble":"Ensemble Prediction"
        })
        st.dataframe(metrics_df.set_index("date"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: “Choropleth Map”
# ─────────────────────────────────────────────────────────────────────────────
elif selection == "Choropleth Map":
    st.title("Predicted Burglary Map")

    # Load precomputed predictions (from test_predictions_fixed.csv)
    results = load_prediction_data()

    # Load & cache the geojson
    geojson = load_geojson()

    # Build Plotly‐Mapbox choropleth
    fig_map = px.choropleth_mapbox(
        results,
        geojson=geojson,
        featureidkey="properties.WD20CD",
        locations="ward_code",
        color="pred_ensemble",  # column name for ensemble predictions
        mapbox_style="carto-positron",
        center={"lat": 51.5, "lon": 0.1},
        zoom=10,
        opacity=0.75,
        color_continuous_scale="Reds",
        labels={"pred_ensemble": "Predicted Burglary Count"},
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    st.plotly_chart(fig_map, use_container_width=True)

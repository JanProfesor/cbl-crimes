import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# DATA & CONFIG
# ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parent.parent
DATA_DIR = ROOT_DIR / "processed"
LOOKUP_FP = ROOT_DIR / "data_preparation" / "z_old" / "lsoa_to_ward_lookup_2020.csv"

_df = pd.read_csv(DATA_DIR / "final_dataset_residential_burglary.csv")
_df["date"] = pd.to_datetime(_df[["year", "month"]].assign(day=1))

lookup = pd.read_csv(LOOKUP_FP)
ward_lookup = (
    lookup[["WD20CD", "WD20NM"]]
    .drop_duplicates()
    .rename(columns={"WD20CD": "ward_code", "WD20NM": "ward_name"})
)
_df = _df.merge(ward_lookup, on="ward_code", how="left")

DOMAIN_SCORES = [
    "crime_score",
    "education_score",
    "employment_score",
    "environment_score",
    "health_score",
    "housing_score",
    "income_score",
]
NUMERIC_COLS = ["burglary_count", "house_price"] + DOMAIN_SCORES

pio.templates.default = "ggplot2"  # pleasant light palette
BLUE = "#1f77b4"

# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, icon: str = "far fa-chart-bar", color: str = "primary") -> dbc.Col:
    """Return a Bootstrap KPI card wrapped in a column."""
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.I(className=f"{icon} fa-2x text-{color}"),
                    html.H6(title, className="mt-2"),
                    html.H4(value, className="fw-bold"),
                ]
            ),
            className="shadow-sm text-center",
        ),
        md=3,
    )

# ────────────────────────────────────────────────────────────────
# DASH APP FACTORY
# ────────────────────────────────────────────────────────────────

def make_dashboard() -> dash.Dash:
    external_stylesheets = [
        dbc.themes.FLATLY,  # Bootswatch theme
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css",  # icons
    ]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = "London Residential Burglaries Insights"

    min_date, max_date = _df["date"].min(), _df["date"].max()

    # --------------- LAYOUT -------------------------------------
    app.layout = dbc.Container(
        fluid=True,
        children=[
            # ─────────── Title ───────────
            dbc.Row(
                dbc.Col(
                    html.H1(
                        "London Residential Burglaries Insights",
                        className="text-center my-4"
                    )
                )
            ),

            # ──────── DatePicker ─────────
            dbc.Row(
                dbc.Col(
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date=min_date,
                        end_date=max_date,
                        display_format="YYYY-MM"
                    ),
                    width="auto",
                ),
                justify="center",
            ),

            # ─────────── KPI ────────────
            dbc.Row(id="kpi-container", className="gy-4 my-3"),

            # ─── Scatter Axis Selectors ──  ← Moved above the scatter plot
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label("Scatter X:", className="fw-bold mb-1"),
                                    dcc.Dropdown(
                                        id="scatter-x",
                                        options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                        value="crime_score",
                                        clearable=False
                                    ),
                                ]
                            ),
                            className="shadow-sm p-3",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label("Scatter Y:", className="fw-bold mb-1"),
                                    dcc.Dropdown(
                                        id="scatter-y",
                                        options=[{"label": c, "value": c} for c in NUMERIC_COLS],
                                        value="house_price",
                                        clearable=False
                                    ),
                                ]
                            ),
                            className="shadow-sm p-3",
                        ),
                        md=3,
                    ),
                ],
                className="gy-4",
            ),

            # ─── Scatter + Bar row ─────
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="scatter-crime-price"), md=6),
                    dbc.Col(dcc.Graph(id="bar-top-wards"), md=6),
                ],
                className="gy-4",
            ),

            # ─── TimeSeries + Parallel ─
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="ts-burglary"), md=6),
                    dbc.Col(dcc.Graph(id="parallel-domains"), md=6),
                ],
                className="gy-4",
            ),
        ],
        style={"background": "#f5f7fa"},
    )

    # --------------- CALLBACK -----------------------------------
    @app.callback(
        [
            Output("kpi-container", "children"),
            Output("ts-burglary", "figure"),
            Output("bar-top-wards", "figure"),
            Output("scatter-crime-price", "figure"),
            Output("parallel-domains", "figure"),
        ],
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("scatter-x", "value"),
            Input("scatter-y", "value"),
        ],
    )
    def update_dashboard(start, end, x_attr, y_attr):
        # 1) slice by date
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) + pd.offsets.MonthEnd(0)
        dff = _df[( _df["date"] >= start_dt ) & ( _df["date"] <= end_dt )]

        # 2) calculate KPIs
        total_burglary = dff["burglary_count"].sum()
        monthly_per_ward = (
            dff
            .groupby(["ward_code", "date"])["burglary_count"]
            .sum()
            .groupby("ward_code")
            .mean()
            .mean()
        )
        ward_top = (
            dff
            .groupby("ward_name")["burglary_count"]
            .sum()
            .nlargest(1)
            .reset_index()
        )
        top_ward_name = ward_top.iloc[0, 0]
        top_ward_val = ward_top.iloc[0, 1]
        avg_crime = dff["crime_score"].mean()

        kpis = [
            kpi_card(
                "Residential Burglaries (Selected Period)",
                f"{total_burglary:,.0f}",
                "fas fa-door-open",
                "danger"
            ),
            kpi_card(
                "Avg Burglaries / Month / Ward",
                f"{monthly_per_ward:.2f}",
                "fas fa-chart-line",
                "primary"
            ),
            kpi_card(
                "Ward with Most Burglaries",
                f"{top_ward_name} ({top_ward_val})",
                "fas fa-map-marker-alt",
                "info"
            ),
            kpi_card(
                "Avg Crime Score",
                f"{avg_crime:.2f}",
                "fas fa-balance-scale",
                "secondary"
            ),
        ]

        # 3) time series line chart
        ts = dff.groupby("date")["burglary_count"].mean().reset_index()
        fig_ts = px.line(
            ts,
            x="date",
            y="burglary_count",
            title="Avg Monthly Residential Burglaries"
        )
        fig_ts.update_traces(line=dict(color=BLUE, width=3))

        # 4) bar chart for top wards
        top10 = (
            dff
            .groupby(["ward_code", "ward_name"])["burglary_count"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        fig_top = px.bar(
            top10,
            x="ward_name",
            y="burglary_count",
            title="Top 10 Wards by Avg Residential Burglaries"
        )
        fig_top.update_traces(marker_color=BLUE)

        # 5) scatter plot + red regression line
        fig_sc = px.scatter(
            dff,
            x=x_attr,
            y=y_attr,
            hover_data=["ward_name"],
            title=f"{y_attr} vs {x_attr}",
            opacity=0.75
        )
        fig_sc.update_traces(marker=dict(color=BLUE))

        if len(dff) > 1:
            x_clean = dff[x_attr].dropna()
            y_clean = dff.loc[x_clean.index, y_attr]
            if len(x_clean) > 1:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_range = slope * x_range + intercept
                # Add regression line last so it sits on top
                fig_sc.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode="lines",
                        name="Trend Line",
                        line=dict(width=2, color="red")
                    )
                )

        # 6) parallel‐coordinates
        parallel_df = (
            dff
            .groupby("ward_name")[DOMAIN_SCORES + ["burglary_count"]]
            .mean()
            .reset_index()
        )
        fig_par = px.parallel_coordinates(
            parallel_df,
            dimensions=DOMAIN_SCORES + ["burglary_count"],
            color="burglary_count",
            color_continuous_scale="OrRd",
            title="IMD Domain Scores vs Avg Residential Burglary",
        )

        # 7) apply transparent background + thin black grid to all
        for fig in (fig_ts, fig_top, fig_sc, fig_par):
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=80, b=40),  # more top margin so titles don't overlap
            )
            fig.update_xaxes(showgrid=True, gridcolor="black", gridwidth=0.5)
            fig.update_yaxes(showgrid=True, gridcolor="black", gridwidth=0.5)

        return kpis, fig_ts, fig_top, fig_sc, fig_par

    return app

# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = make_dashboard()
    app.run(debug=True, port=8050)

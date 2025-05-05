import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess data
df = pd.read_csv('processed/final_dataset.csv')
df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))

# Load ward lookup for names
lookup = pd.read_csv('lsoa_to_ward_lookup_2020.csv')
ward_lookup = lookup[['WD20CD','WD20NM']].drop_duplicates().rename(columns={'WD20CD':'ward_code','WD20NM':'ward_name'})
# Merge ward names
df = df.merge(ward_lookup, on='ward_code', how='left')

# Attributes for scatter dropdown
domain_scores = [
    'crime_score','education_score','employment_score',
    'environment_score','health_score','housing_score','income_score'
]
numeric_cols = ['burglary_count','house_price'] + domain_scores

# Build Dash app
def make_dashboard():
    app = dash.Dash(__name__)
    app.title = 'London Ward Insights'

    min_date, max_date = df['date'].min(), df['date'].max()

    app.layout = html.Div([
        html.H1('London Ward Crime & Housing Insights', style={'textAlign':'center'}),
        html.Div([
            html.Label('Select Date Range:'),
            dcc.DatePickerRange(
                id='date-range', start_date=min_date, end_date=max_date, display_format='YYYY-MM'
            )
        ], style={'textAlign':'center','padding':'10px'}),

        html.Div([
            html.Div(id='kpi-container', style={'flex':'3','display':'flex','justifyContent':'space-around'}),
            html.Div([
                html.Label('Scatter: X and Y:'),
                dcc.Dropdown(id='scatter-x', options=[{'label':c,'value':c} for c in numeric_cols], value='crime_score'),
                dcc.Dropdown(id='scatter-y', options=[{'label':c,'value':c} for c in numeric_cols], value='house_price')
            ], style={'flex':'1','padding':'20px'})
        ], style={'display':'flex','alignItems':'center'}),

        html.Div([
            dcc.Graph(id='ts-burglary', style={'flex':'1'}),
            dcc.Graph(id='bar-top-wards', style={'flex':'1'})
        ], style={'display':'flex'}),

        html.Div([
            dcc.Graph(id='scatter-crime-price', style={'flex':'1'}),
            dcc.Graph(id='parallel-domains', style={'flex':'1'})
        ], style={'display':'flex'})
    ])

    @app.callback(
        [Output('kpi-container','children'),
         Output('ts-burglary','figure'),
         Output('bar-top-wards','figure'),
         Output('scatter-crime-price','figure'),
         Output('parallel-domains','figure')],
        [Input('date-range','start_date'), Input('date-range','end_date'),
         Input('scatter-x','value'), Input('scatter-y','value')]
    )
    def update_dashboard(start, end, x_attr, y_attr):
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) + pd.offsets.MonthEnd(0)
        dff = df[(df['date']>=start_dt)&(df['date']<=end_dt)]

        # KPIs
        total_burglary = dff['burglary_count'].sum()
        avg_burglary = dff.groupby('ward_code')['burglary_count'].mean().mean()
        avg_price = dff['house_price'].mean()
        avg_crime = dff['crime_score'].mean()
        kpis = [
            html.Div([html.H3('Total Burglaries'), html.P(f'{total_burglary:.0f}')]),
            html.Div([html.H3('Avg Burglaries/Ward'), html.P(f'{avg_burglary:.1f}')]),
            html.Div([html.H3('Avg House Price'), html.P(f'£{avg_price:,.0f}')]),
            html.Div([html.H3('Avg Crime Score'), html.P(f'{avg_crime:.2f}')])
        ]

        # Time series
        ts = dff.groupby('date')['burglary_count'].mean().reset_index()
        fig_ts = px.line(ts, x='date', y='burglary_count', title='Avg Monthly Burglary Count')

        # Top wards by avg burglary (names)
        top10 = (
            dff.groupby(['ward_code','ward_name'])['burglary_count']
            .mean().nlargest(10).reset_index()
        )
        fig_top = px.bar(top10, x='ward_name', y='burglary_count', title='Top 10 Wards by Avg Burglaries')

        # Scatter with regression line
        fig_sc = px.scatter(dff, x=x_attr, y=y_attr, hover_data=['ward_name'], title=f'{y_attr} vs {x_attr}')
        # Regression
        if len(dff) > 1:
            x_clean = dff[x_attr].dropna()
            y_clean = dff[y_attr].loc[x_clean.index]
            if len(x_clean) > 1:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
                x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_range = slope * x_range + intercept
                fig_sc.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Trend Line'))

        # Parallel coordinates
        parallel_df = dff.groupby('ward_name')[domain_scores + ['burglary_count']].mean().reset_index()
        fig_par = px.parallel_coordinates(
            parallel_df,
            dimensions=domain_scores + ['burglary_count'],
            color='burglary_count', color_continuous_scale='OrRd',
            title='IMD Domain Scores vs Avg Burglary per Ward'
        )

        return kpis, fig_ts, fig_top, fig_sc, fig_par

    return app

if __name__=='__main__':
    app = make_dashboard()
    app.run(debug=True, port=8050)

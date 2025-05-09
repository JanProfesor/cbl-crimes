import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess both datasets
df1 = pd.read_csv('processed/final_dataset.csv')
df2 = pd.read_csv(r"C:/Users/matei/Downloads/final_dataset1.csv") 
for df in (df1, df2):
    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))

# Load ward lookup for names
lookup = pd.read_csv('lsoa_to_ward_lookup_2020.csv')
ward_lookup = lookup[['WD20CD','WD20NM']].drop_duplicates()
ward_lookup.columns = ['ward_code','ward_name']
df1 = df1.merge(ward_lookup, on='ward_code', how='left')
df2 = df2.merge(ward_lookup, on='ward_code', how='left')

# Sidebar filters
st.sidebar.header('Filters')
min_date = min(df1['date'].min(), df2['date'].min())
max_date = max(df1['date'].max(), df2['date'].max())
start_date = st.sidebar.date_input('Start date', min_date)
end_date = st.sidebar.date_input('End date', max_date)
if start_date > end_date:
    st.sidebar.error('Error: End date must fall after start date.')
else:
    df1 = df1[(df1['date'] >= pd.to_datetime(start_date)) & (df1['date'] <= pd.to_datetime(end_date))]
    df2 = df2[(df2['date'] >= pd.to_datetime(start_date)) & (df2['date'] <= pd.to_datetime(end_date))]

# Scatter axis selection
attributes = [
    'burglary_count','house_price',
    'crime_score','education_score','employment_score',
    'environment_score','health_score','housing_score','income_score'
]
x_attr = st.sidebar.selectbox('X Axis', attributes, index=attributes.index('crime_score'))
y_attr = st.sidebar.selectbox('Y Axis', attributes, index=attributes.index('house_price'))

st.title('London Ward Crime & Housing Insights (Dual Datasets)')

# Time Series: Avg Monthly Burglary
st.subheader('Avg Monthly Burglary Count')
col_ts1, col_ts2 = st.columns(2)
# Dataset 1
ts1 = df1.groupby('date')['burglary_count'].mean().reset_index()
fig_ts1 = px.line(ts1, x='date', y='burglary_count', title='Dataset 1: Avg Monthly Burglary')
col_ts1.plotly_chart(fig_ts1, use_container_width=True)
# Dataset 2
ts2 = df2.groupby('date')['burglary_count'].mean().reset_index()
fig_ts2 = px.line(ts2, x='date', y='burglary_count', title='Dataset 2: Avg Monthly Burglaries')
col_ts2.plotly_chart(fig_ts2, use_container_width=True)

# Top 10 Wards by Avg Burglaries
st.subheader('Top 10 Wards by Avg Burglaries')
col_top1, col_top2 = st.columns(2)
# Dataset 1
top10_1 = df1.groupby(['ward_code','ward_name'])['burglary_count'].mean().nlargest(10).reset_index()
fig_top1 = px.bar(top10_1, x='ward_name', y='burglary_count', title='Dataset 1: Top 10 Wards', labels={'ward_name':'Ward','burglary_count':'Avg Burglaries'})
col_top1.plotly_chart(fig_top1, use_container_width=True)
# Dataset 2
top10_2 = df2.groupby(['ward_code','ward_name'])['burglary_count'].mean().nlargest(10).reset_index()
fig_top2 = px.bar(top10_2, x='ward_name', y='burglary_count', title='Dataset 2: Top 10 Wards', labels={'ward_name':'Ward','burglary_count':'Avg Burglaries'})
col_top2.plotly_chart(fig_top2, use_container_width=True)

# Scatter Plot: Selected Attributes
st.subheader(f'{y_attr.replace("_"," ").title()} vs {x_attr.replace("_"," ").title()}')
col_sc1, col_sc2 = st.columns(2)
# Dataset 1
fig_sc1 = px.scatter(df1, x=x_attr, y=y_attr, hover_data=['ward_name'], title=f'Dataset 1: {y_attr.title()} vs {x_attr.title()}')
clean1 = df1[[x_attr,y_attr]].dropna()
if len(clean1) > 1:
    slope, intercept = np.polyfit(clean1[x_attr], clean1[y_attr], 1)
    x_range = np.array([clean1[x_attr].min(), clean1[x_attr].max()])
    y_range = slope * x_range + intercept
    fig_sc1.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Trend Line'))
col_sc1.plotly_chart(fig_sc1, use_container_width=True)
# Dataset 2
fig_sc2 = px.scatter(df2, x=x_attr, y=y_attr, hover_data=['ward_name'], title=f'Dataset 2: {y_attr.title()} vs {x_attr.title()}')
clean2 = df2[[x_attr,y_attr]].dropna()
if len(clean2) > 1:
    slope2, intercept2 = np.polyfit(clean2[x_attr], clean2[y_attr], 1)
    x_range2 = np.array([clean2[x_attr].min(), clean2[x_attr].max()])
    y_range2 = slope2 * x_range2 + intercept2
    fig_sc2.add_trace(go.Scatter(x=x_range2, y=y_range2, mode='lines', name='Trend Line'))
col_sc2.plotly_chart(fig_sc2, use_container_width=True)

# Parallel Coordinates: IMD Domains vs Avg Burglary per Ward
st.subheader('IMD Domain Scores vs Avg Burglary per Ward')
col_par1, col_par2 = st.columns(2)
# Dataset 1
parallel1 = df1.groupby('ward_name')[['crime_score','education_score','employment_score',
                                       'environment_score','health_score','housing_score',
                                       'income_score','burglary_count']].mean().reset_index()
fig_par1 = px.parallel_coordinates(parallel1,
                                   dimensions=['crime_score','education_score','employment_score',
                                               'environment_score','health_score','housing_score',
                                               'income_score','burglary_count'],
                                   color='burglary_count', color_continuous_scale='OrRd',
                                   title='Dataset 1: IMD vs Avg Burgary')
col_par1.plotly_chart(fig_par1, use_container_width=True)
# Dataset 2
parallel2 = df2.groupby('ward_name')[['crime_score','education_score','employment_score',
                                       'environment_score','health_score','housing_score',
                                       'income_score','burglary_count']].mean().reset_index()
fig_par2 = px.parallel_coordinates(parallel2,
                                   dimensions=['crime_score','education_score','employment_score',
                                               'environment_score','health_score','housing_score',
                                               'income_score','burglary_count'],
                                   color='burglary_count', color_continuous_scale='OrRd',
                                   title='Dataset 2: IMD vs Avg Burgary')
col_par2.plotly_chart(fig_par2, use_container_width=True)

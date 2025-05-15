import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess data
df = pd.read_csv('processed/final_dataset_residential_burglary.csv')
df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))

# Load ward lookup for names
lookup = pd.read_csv('lsoa_to_ward_lookup_2020.csv')
ward_lookup = lookup[['WD20CD','WD20NM']].drop_duplicates()
ward_lookup.columns = ['ward_code','ward_name']
df = df.merge(ward_lookup, on='ward_code', how='left')

# Sidebar filters
st.sidebar.header('Filters')
min_date, max_date = df['date'].min(), df['date'].max()
start_date = st.sidebar.date_input('Start date', min_date)
end_date = st.sidebar.date_input('End date', max_date)
if start_date > end_date:
    st.sidebar.error('Error: End date must fall after start date.')
else:
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

# Scatter axis selection
attributes = ['burglary_count','house_price',
              'crime_score','education_score','employment_score',
              'environment_score','health_score','housing_score','income_score']
x_attr = st.sidebar.selectbox('X Axis', attributes, index=attributes.index('crime_score'))
y_attr = st.sidebar.selectbox('Y Axis', attributes, index=attributes.index('house_price'))


st.title('London Ward Residential Burglary Analysis')


total_burg = df['burglary_count'].sum()
avg_burg_ward = df.groupby('ward_code')['burglary_count'].mean().mean()
avg_price = df['house_price'].mean()
avg_crime = df['crime_score'].mean()
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Burglaries', f'{int(total_burg):,}')
col2.metric('Avg Burglaries/Ward', f'{avg_burg_ward:.2f}')
col3.metric('Avg House Price', f'£{avg_price:,.0f}')
col4.metric('Avg Crime Score', f'{avg_crime:.2f}')

# Time Series: Avg Monthly Burglary
ts = df.groupby('date')['burglary_count'].mean().reset_index()
fig_ts = px.line(ts, x='date', y='burglary_count', title='Avg Monthly Residential Burglary Count')
st.plotly_chart(fig_ts, use_container_width=True)


top10 = df.groupby(['ward_code','ward_name'])['burglary_count']\
           .mean().nlargest(10).reset_index()
fig_top = px.bar(top10, x='ward_name', y='burglary_count',
                 title='Top 10 Wards by Avg Residential Burglaries',
                 labels={'ward_name':'Ward','burglary_count':'Avg Burglaries'})
st.plotly_chart(fig_top, use_container_width=True)


fig_sc = px.scatter(df, x=x_attr, y=y_attr, hover_data=['ward_name'],
                    title=f'{y_attr.replace("_"," ").title()} vs {x_attr.replace("_"," ").title()}')
if len(df) > 1:
    clean = df[[x_attr,y_attr]].dropna()
    if len(clean) > 1:
        slope, intercept = np.polyfit(clean[x_attr], clean[y_attr], 1)
        x_range = np.array([clean[x_attr].min(), clean[x_attr].max()])
        y_range = slope * x_range + intercept
        fig_sc.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Trend Line'))
st.plotly_chart(fig_sc, use_container_width=True)

# NEW SECTION: Correlation Heatmap for Factors Related to Burglary
st.subheader("Correlation Analysis: Factors Related to Residential Burglary")

# Create correlation matrix
corr_cols = ['burglary_count', 'house_price', 'crime_score', 'education_score', 
             'employment_score', 'environment_score', 'health_score', 
             'housing_score', 'income_score']
             
corr_df = df.groupby('ward_name')[corr_cols].mean().reset_index()
corr_matrix = corr_df[corr_cols].corr().round(2)

# Create heatmap
fig_heatmap = px.imshow(
    corr_matrix,
    x=corr_cols,
    y=corr_cols,
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    title="Correlation of Residential Burglary with Socioeconomic Factors"
)

fig_heatmap.update_layout(
    xaxis_title="", 
    yaxis_title="",
    xaxis={'side': 'bottom'}
)

# Add text annotations
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.index)):
        fig_heatmap.add_annotation(
            x=i, y=j,
            text=str(corr_matrix.iloc[j, i]),
            showarrow=False,
            font=dict(color="white" if abs(corr_matrix.iloc[j, i]) > 0.5 else "black")
        )

st.plotly_chart(fig_heatmap, use_container_width=True)

# NEW SECTION: Seasonal Analysis of Burglaries
st.subheader("Seasonal Analysis of Residential Burglaries")

# Extract month and calculate average burglaries by month
df['month_name'] = df['date'].dt.strftime('%B')
monthly_burglaries = df.groupby('month_name')['burglary_count'].mean().reset_index()

# Define month order for proper sorting
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_burglaries['month_num'] = monthly_burglaries['month_name'].apply(lambda x: month_order.index(x))
monthly_burglaries = monthly_burglaries.sort_values('month_num')

# Create bar chart
fig_seasonal = px.bar(
    monthly_burglaries, 
    x='month_name', 
    y='burglary_count',
    category_orders={"month_name": month_order},
    title="Average Residential Burglaries by Month",
    labels={'month_name': 'Month', 'burglary_count': 'Average Burglary Count'}
)

# Add a horizontal line for the overall average
overall_avg = monthly_burglaries['burglary_count'].mean()
fig_seasonal.add_hline(
    y=overall_avg,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Overall Average: {overall_avg:.2f}",
    annotation_position="top right"
)

st.plotly_chart(fig_seasonal, use_container_width=True)
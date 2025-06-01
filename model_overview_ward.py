import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Dashboard to visualize model performance over time by ward name


@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Expect either 'ward_name' or only 'ward' (code)
    if 'ward_name' not in df.columns:
        df['ward_name'] = df['ward'].astype(str)
    
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    return df


def main():
    st.title("Model Performance Dashboard")
    st.markdown("Choose a ward to see actual vs. predicted burglary counts over time.")

    data = load_data("test_predictions_final.csv")

    ward_list = sorted(data['ward_name'].unique())
    selected_ward = st.sidebar.selectbox("Select Ward", ward_list)

    ward_df = data[data['ward_name'] == selected_ward].sort_values('date')

    st.subheader(f"Time Series: {selected_ward}")
    if ward_df.empty:
        st.write("No data available for this ward.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ward_df['date'], ward_df['actual'], label='Actual', marker='o')
    ax.plot(ward_df['date'], ward_df['pred_ensemble'], label='Predicted', marker='x')
    ax.set_xlabel('Date')
    ax.set_ylabel('Burglary Count')
    ax.set_title(f"Actual vs. Ensemble Predicted for Ward: {selected_ward}")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Numeric Metrics Over Time")
    metrics_df = ward_df[['date', 'actual', 'pred_tabnet', 'pred_xgboost', 'pred_ensemble']].copy()
    metrics_df = metrics_df.rename(columns={
        'pred_tabnet': 'TabNet Prediction',
        'pred_xgboost': 'XGBoost Prediction',
        'pred_ensemble': 'Ensemble Prediction'
    })
    st.dataframe(metrics_df.set_index('date'))

if __name__ == "__main__":
    main()

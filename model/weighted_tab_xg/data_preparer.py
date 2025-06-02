import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreparer:
    def __init__(self, csv_path, target):
        self.df = pd.read_csv(csv_path)
        self.target = target

    def preprocess(self):
        df = self.df.copy()
        # Create a date column for sorting
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values(['ward_code', 'date'])

        # Create lag features
        for lag in (1, 2, 3, 6, 12):
            df[f'{self.target}_lag_{lag}'] = df.groupby('ward_code')[self.target].shift(lag)

        # Rolling statistics
        for window in [3, 6, 12]:
            df[f'{self.target}_roll_mean_{window}'] = (
                df.groupby('ward_code')[self.target]
                  .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f'{self.target}_roll_std_{window}'] = (
                df.groupby('ward_code')[self.target]
                  .transform(lambda x: x.rolling(window, min_periods=1).std())
            )

        # Cyclical encoding for month
        df['month_num'] = df['date'].dt.month
        df['year_num'] = df['date'].dt.year
        ang = 2 * np.pi * df['month_num'] / 12
        df['month_sin'] = np.sin(ang)
        df['month_cos'] = np.cos(ang)

        # Drop rows with NaNs from lag/rolling
        df = df.dropna().reset_index(drop=True)

        # Extract original identifiers
        df['ward_code_orig'] = df['ward_code']
        df['year_orig'] = df['year']
        df['month_orig'] = df['month']
        df['actual'] = df[self.target]

        # Prepare features DataFrame
        features_to_drop = [self.target, 'date', 'year', 'month']
        X = df.drop(columns=features_to_drop).copy()

        # Encode ward_code as categorical codes
        X['ward_code'] = X['ward_code'].astype('category').cat.codes
        cat_idxs = [X.columns.get_loc('ward_code')]
        cat_dims = [X['ward_code'].nunique()]

        # Identify numerical columns
        num_cols = [col for col in X.columns if col not in ['ward_code', 'ward_code_orig', 'year_orig', 'month_orig', 'actual']]

        # Scale numerical features
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Extract target array
        y = df['actual'].values

        return X, y, cat_idxs, cat_dims, df
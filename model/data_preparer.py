import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreparer:
    def __init__(self, csv_path, target):
        self.df = pd.read_csv(csv_path)
        self.target = target

    def preprocess(self):
        df = self.df
        df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
        df = df.sort_values(['ward_code','date'])
        for lag in (1, 2, 3, 6, 12):
            df[f'{self.target}_lag_{lag}'] = df.groupby('ward_code')[self.target].shift(lag)
        for window in [3, 6, 12]:
            df[f'{self.target}_roll_mean_{window}'] = df.groupby('ward_code')[self.target].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{self.target}_roll_std_{window}'] = df.groupby('ward_code')[self.target].transform(
                lambda x: x.rolling(window, min_periods=1).std())
        df['month_num'] = df['date'].dt.month
        df['year_num'] = df['date'].dt.year
        ang = 2*np.pi*df['month_num']/12
        df['month_sin'] = np.sin(ang)
        df['month_cos'] = np.cos(ang)
        df.dropna(inplace=True)
        features_to_drop = [self.target, 'date', 'year', 'month']
        X = df.drop(columns=features_to_drop)
        X['ward_code'] = X['ward_code'].astype('category').cat.codes
        cat_idxs = [X.columns.get_loc('ward_code')]
        cat_dims = [X['ward_code'].nunique()]
        num_cols = [col for col in X.columns if col != 'ward_code']
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        y = df[self.target].values
        return X, y, cat_idxs, cat_dims
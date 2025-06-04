import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import holidays


class DataPreparerNoLeakage:
    """
    Data preparer that avoids temporal leakage by only using past information
    for feature engineering within each time period.
    """

    def __init__(self, csv_path: str, target: str):
        self.df = pd.read_csv(csv_path)
        self.target = target
        adj_df = pd.read_csv("ward_adjacency.csv")
        self.neighbors = adj_df.groupby("ward_code")["neighbor_code"].apply(list).to_dict()
        years = range(self.df["year"].min(), self.df["year"].max() + 1)
        self.uk_holidays = holidays.UnitedKingdom(years=years)

    

    def create_features_no_leakage(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag + rolling features using only data up to current point,
        but compute neighbor_count_lag1 via a vectorized merge instead of a row‐loop.
        """
        df = df_subset.copy()
        # 1) Build date, sort (as before)
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        df = df.sort_values(["ward_code", "date"]).reset_index(drop=True)

        # 2) BASIC lag features
        for lag in (1, 2, 3, 6, 12):
            df[f"{self.target}_lag_{lag}"] = (
                df.groupby("ward_code")[self.target]
                .shift(lag)
            )

        # 3) Rolling statistics
        for window in (3, 6, 12):
            roll_mean = (
                df.groupby("ward_code")[self.target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            roll_std = (
                df.groupby("ward_code")[self.target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
            )
            df[f"{self.target}_roll_mean_{window}"] = roll_mean
            df[f"{self.target}_roll_std_{window}"]  = roll_std

        # ─── NEW: HOLIDAY FLAG and WEEKEND‐PCT ─────────────────────────────────
        unique_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)

        # B) Build a small DataFrame of holiday/weekend features per unique date
        month_feats = pd.DataFrame({"date": unique_dates})

        # Function to compute (is_holiday, weekend_pct) for a given Timestamp ts
        def month_flag(ts):
            year, month = ts.year, ts.month
            start = pd.Timestamp(year, month, 1)
            end   = start + pd.offsets.MonthEnd(0)
            all_days = pd.date_range(start, end, freq="D")

            # Check holiday membership only once for the entire month
            has_hol = any((day.date() in self.uk_holidays) for day in all_days)
            wk_pct  = (all_days.weekday >= 5).sum() / len(all_days)
            return pd.Series({
                "is_holiday_month": int(has_hol),
                "pct_weekend_days": wk_pct
            })

        # Apply month_flag to each unique date (M rows, not N)
        month_feats[["is_holiday_month", "pct_weekend_days"]] = month_feats["date"].apply(month_flag)

        # C) Merge these two new columns back into df on “date”
        df = df.merge(month_feats, on="date", how="left")
        
        
        # ─────── NEW: NEIGHBOR‐LAG FEATURE (vectorized merge) ────────────────

        # “Explode” the neighbors dict into a DataFrame of (ward_code, neighbor_code) rows
        #  Example: if self.neighbors["A"] = ["B","C"], you get two rows: ("A","B") and ("A","C")
        adj_df = (
            pd.DataFrame([(ward, nbr) 
                        for ward, nbrs in self.neighbors.items() 
                        for nbr in nbrs],
                        columns=["ward_code", "neighbor_code"])
        )

        
        prev_df = df[["ward_code", "date", self.target]].copy()
        prev_df["date"] = prev_df["date"] - pd.offsets.MonthBegin(1)
        prev_df = prev_df.rename(columns={self.target: "neighbor_count"})

        prev_df = prev_df.rename(columns={"ward_code": "neighbor_code", "date": "prev_date"})
        
        df_for_merge = df[["ward_code", "date"]].copy()
        df_for_merge = df_for_merge.drop_duplicates()  
        #    This is each distinct (ward, date). We will attach neighbor info to each row.

        
        merged_1 = pd.merge(
            df_for_merge,
            adj_df,
            on="ward_code",
            how="left"
        )
        

        
        merged_2 = pd.merge(
            merged_1,
            prev_df,
            left_on=["neighbor_code", "date"],
            right_on=["neighbor_code", "prev_date"],
            how="left"
        )
        
        neighbor_avg = (
            merged_2
            .groupby(["ward_code", "date"])["neighbor_count"]
            .mean()
            .reset_index()
            .rename(columns={"neighbor_count": "neighbor_count_lag1"})
        )
        #    neighbor_avg = DataFrame with columns ['ward_code','date','neighbor_count_lag1']
        #    If a ward has no neighbors or no data in prev_month, mean() will be NaN, so fill with 0:
        neighbor_avg["neighbor_count_lag1"] = neighbor_avg["neighbor_count_lag1"].fillna(0.0)

        # E) Finally, merge neighbor_avg back into the original df on ['ward_code','date']
        df = pd.merge(
            df,
            neighbor_avg,
            on=["ward_code", "date"],
            how="left"
        )
        #    Any ward‐date that had no neighbor row will have NaN→0.0 after fillna above

        # ───────────────────────────────────────────────────────────────────────
        # 4) (Optional) Drop any rows with missing first‐lag as before
        df = df.dropna(subset=[f"{self.target}_lag_1"]).reset_index(drop=True)

        return df


    def preprocess_split_aware(self, train_end_date=None):
        """
        Preprocess data with proper temporal split to avoid leakage.
        """
        df = self.df.copy()

        # 1) Create 'date' for sorting
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        
        # 2) Sort by date first, then ward
        df = df.sort_values(["date", "ward_code"]).reset_index(drop=True)
        
        # 3) Determine split point (70% of timeline)
        if train_end_date is None:
            unique_dates = sorted(df["date"].unique())
            split_idx = int(len(unique_dates) * 0.7)
            train_end_date = unique_dates[split_idx]
        
        # 4) Split data temporally
        train_mask = df["date"] <= train_end_date
        df_train_raw = df[train_mask].copy()
        df_test_raw = df[~train_mask].copy()
        
        # 5) Create features separately for train and test (avoiding leakage)
        # For training: only use training data to compute features
        df_train = self.create_features_no_leakage(df_train_raw)
        
        # For test: use ALL data up to each test point (train + previous test points)
        # This simulates real-world scenario where you have historical data
        df_test_list = []
        test_dates = sorted(df_test_raw["date"].unique())
        
        for test_date in test_dates:
            # For each test date, use all data up to that point
            historical_data = df[df["date"] <= test_date].copy()
            features_data = self.create_features_no_leakage(historical_data)
            # But only keep the rows for this specific test date
            test_rows = features_data[features_data["date"] == test_date]
            if not test_rows.empty:
                df_test_list.append(test_rows)
        
        if df_test_list:
            df_test = pd.concat(df_test_list, ignore_index=True)
        else:
            df_test = pd.DataFrame()

        # 6) Add cyclical features
        for df_subset in [df_train, df_test]:
            if not df_subset.empty:
                df_subset["month_num"] = df_subset["date"].dt.month
                df_subset["year_num"] = df_subset["date"].dt.year
                ang = 2 * np.pi * df_subset["month_num"] / 12
                df_subset["month_sin"] = np.sin(ang)
                df_subset["month_cos"] = np.cos(ang)

        # 7) Drop NaN rows (from lag features)
        df_train = df_train.dropna().reset_index(drop=True)
        df_test = df_test.dropna().reset_index(drop=True)
        
        # 8) Prepare identifier columns
        for df_subset in [df_train, df_test]:
            if not df_subset.empty:
                df_subset["ward_code_orig"] = df_subset["ward_code"]
                df_subset["year_orig"] = df_subset["year"]
                df_subset["month_orig"] = df_subset["month"]
                df_subset["actual"] = df_subset[self.target]

        return df_train, df_test, train_end_date

    def prepare_features(self, df_subset):
        """Extract features from a dataframe subset"""
        if df_subset.empty:
            return pd.DataFrame(), np.array([])
            
        # Drop target and identifier columns
        drop_cols = [self.target, "date", "year", "month"]
        X = df_subset.drop(columns=drop_cols, errors="ignore")
        y = df_subset["actual"].values
        
        return X, y
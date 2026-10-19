import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class LOFPointAnomalyDetector:
    def __init__(self, neighbours=14, contamination=0.01):
        self.neighbours = neighbours
        self.contamination = contamination

    def detect(self, series):

        # Make series 2D
        X = series.dropna().values.reshape(-1,1)

        # Inicjalization
        clf = LocalOutlierFactor(n_neighbors=self.neighbours, contamination=self.contamination)
        y_pred = clf.fit_predict(X)

        X_scores = clf.negative_outlier_factor_

        anomalies_flag = pd.Series(False, index=series.index)
        anomalies_flag.loc[series.dropna().index] = (y_pred == -1)

        lof_scores = pd.Series(np.nan, index=series.index)
        lof_scores.loc[series.dropna().index] = X_scores

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} LOF point anomalies")

        return anomalies_flag.values,lof_scores.values

class LOFColectiveAbomalyDetector:
    def __init__(self, window_size=14, neighbours=14, contamination=0.01):
        self.window_size = window_size
        self.neighbours = neighbours
        self.contamination = contamination

    def detect(self, series):

        # Roling sum 
        metric = series.rolling(window=self.window_size).sum()
        series_rolling = metric.dropna()

        # Make series 2D
        X = series_rolling.values.reshape(-1,1)

        # Inicjalization
        clf = LocalOutlierFactor(n_neighbors=self.neighbours, contamination=self.contamination)
        y_pred = clf.fit_predict(X)

        X_scores = clf.negative_outlier_factor_

        anomalies_flags = pd.Series(False, index=series.index)
        anomalies_flags.loc[series_rolling.dropna().index] = (y_pred == -1)

        lof_scores = pd.Series(np.nan, index=series.index)
        lof_scores.loc[series_rolling.dropna().index] = X_scores

        anomalies_flag = pd.Series(False, index=series.index)
        for idx in anomalies_flags[anomalies_flags == True].index:
            idx_num = series.index.get_loc(idx)
            window_start = max(0, idx_num - self.window_size + 1)
            window_end = idx_num + 1
            anomalies_flag.iloc[window_start:window_end] = True
        
        anomalies_num = anomalies_flags.sum()
        print(f"[SUCCESS] Detected {anomalies_num} LOF colective anomalies")

        return anomalies_flag.values,lof_scores.values

class LOFContextualAnomalyDetector:
    def __init__(self, neighbours=14, contamination=0.01):
        self.neighbours = neighbours
        self.contamination = contamination

    def detect(self, series, context_series):

        # Conacat series and context series
        context = pd.concat([series, context_series], axis=1).dropna()

        X = context.values

        # Inicjalization
        clf = LocalOutlierFactor(n_neighbors=self.neighbours, contamination=self.contamination)
        y_pred = clf.fit_predict(X)

        X_scores = clf.negative_outlier_factor_

        anomalies_flag = pd.Series(False, index=series.index)
        anomalies_flag.loc[context.dropna().index] = (y_pred == -1)

        lof_scores = pd.Series(np.nan, index=series.index)
        lof_scores.loc[context.dropna().index] = X_scores

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} LOF point anomalies")

        return anomalies_flag.values,lof_scores.values

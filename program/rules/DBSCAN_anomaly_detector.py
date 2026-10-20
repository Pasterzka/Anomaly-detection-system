import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DBSCANPointAnomalyDetector:
    def __init__(self, eps=0.5, min_sample=5):
        self.eps = eps
        self.min_sample = min_sample

    def detect(self, series):

        # Make series 2D
        X = series.dropna().values.reshape(-1,1)
        X_scaled = StandardScaler().fit_transform(X)

        # Inicjalization
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample)
        y_pred = dbscan.fit_predict(X_scaled)


        anomalies_flag = pd.Series(False, index=series.index)
        anomalies_flag.loc[series.dropna().index] = (y_pred == -1)

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} DBSCAN point anomalies")

        return anomalies_flag.values

class DBSCANCollectiveAnomalyDetector:
    def __init__(self, window_size=14, eps=5, min_sample=5):
        self.window_size = window_size
        self.eps = eps
        self.min_sample = min_sample

    def detect(self, series):

        # Roling sum
        metric = series.rolling(window=self.window_size).sum()
        series_rolling = metric.dropna()

        # Make series 2D
        X = series_rolling.values.reshape(-1,1)
        X_scaled = StandardScaler().fit_transform(X)

        # Inicjalization
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample)
        y_pred = dbscan.fit_predict(X_scaled)

        anomalies_flags = pd.Series(False, index=series.index)
        anomalies_flags.loc[series_rolling.dropna().index] = (y_pred == -1)

        anomalies_flag = pd.Series(False, index=series.index)
        for idx in anomalies_flags[anomalies_flags == True].index:
            idx_num = series.index.get_loc(idx)
            window_start = max(0, idx_num - self.window_size + 1)
            window_end = idx_num + 1
            anomalies_flag.iloc[window_start:window_end] = True

        anomalies_num = anomalies_flags.sum()
        print(f"[SUCCESS] Detected {anomalies_num} DBSCAN colective anomalies")

        return anomalies_flag.values

class DBSCANContextualAnomalyDetector:
    def __init__(self, eps=0.5, min_sample=5):
        self.eps = eps
        self.min_sample = min_sample

    def detect(self, series, context_series):

        context = pd.concat([series, context_series], axis=1).dropna()

        # Make series 2D        
        X = context.values
        X_scaled = StandardScaler().fit_transform(X)
    
        # Inicjalization
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample)
        y_pred = dbscan.fit_predict(X_scaled)
    
    
        anomalies_flag = pd.Series(False, index=series.index)
        anomalies_flag.loc[context.dropna().index] = (y_pred == -1)
    
        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} DBSCAN point anomalies")
    
        return anomalies_flag.values


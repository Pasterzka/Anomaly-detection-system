import numpy as np
import pandas as pd

class newCollectiveAnomalyDetector:
    def __init__(self, method='3sigma', multiplier=3.0, window_size=14, baseline_window=50):
        self.method = method
        self.multiplier = multiplier
        self.window_size= window_size
        self.baseline_window = window_size * 7 # time of 7 lates moving window

    def detect(self, series):

        print("[INFO] Detecting collectiv anomalies...")

        # Sum of small group
        group_metric = series.rolling(window=self.window_size).sum()

        if self.method == '3sigma':
            roling_mean = group_metric.rolling(window=self.baseline_window).mean().shift(1)
            roling_std = group_metric.rolling(window=self.baseline_window).std().shift(1)
        
            lower_bound = roling_mean - roling_std * self.multiplier
            upper_bound = roling_mean + roling_std * self.multiplier
        
        elif self.method == 'iqr':
            q1 = group_metric.rolling(window=self.baseline_window).quantile(0.25).shift(1)
            q3 = group_metric.rolling(window=self.baseline_window).quantile(0.75).shift(1)
        
            iqr = q3 - q1
        
            lower_bound = q1 - iqr * self.multiplier
            upper_bound = q3 + iqr * self.multiplier

        anomalies_windows = (group_metric > upper_bound) | (group_metric < lower_bound)

        # all group is a anomally
        anomalies_flag = pd.Series(False, index=series.index)
        for idx in anomalies_windows[anomalies_windows==True].index:
            idx_num = series.index.get_loc(idx)
            window_start = max(0, idx_num - self.window_size + 1)
            window_end = idx_num + 1
            anomalies_flag.iloc[window_start:window_end] = True

        anomalies_num = anomalies_windows.sum()
        print(f"[SUCCESS] Detected {anomalies_num} {self.method} collective anomalies")

        return anomalies_flag.fillna(False).values


import numpy as np
import pandas as pd

# Contextual anomaly detector
class ContextualAnomalyDetector:
    def __init__(self, window_size=14, multiplier=3.0, method="3sigma"):
        self.window_size = window_size
        self.method = method
        self.multiplier = multiplier

    # Nothing here -> context developed dynamiclly
    def fit(self, trein_series):
        pass

    def detect(self, test_series):
        if self.method == '3sigma':
            # Calculate local mean and std 
            rolling_mean = test_series.rolling(window=self.window_size).mean()
            rolling_std = test_series.rolling(window=self.window_size).std()
            lower_bound = rolling_mean - self.multiplier * rolling_std
            upper_bound = rolling_mean + self.multiplier * rolling_std

        elif self.method == 'iqr':
            # Calculate local Q1 and Q3
            rolling_q1 = test_series.rolling(window=self.window_size).quantile(0.25)
            rolling_q3 = test_series.rolling(window=self.window_size).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            lower_bound = rolling_q1 - self.multiplier * rolling_iqr
            upper_bound = rolling_q3 + self.multiplier * rolling_iqr

        anomalies_flags = (test_series < lower_bound) | (test_series > upper_bound)
        anomalies_flags = anomalies_flags.fillna(False)

        return anomalies_flags.values
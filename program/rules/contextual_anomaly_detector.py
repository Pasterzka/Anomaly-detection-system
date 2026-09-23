import numpy as np
import pandas as pd

# Contextual anomaly detector
class ContextualAnomalyDetector:
    def __init__(self, window_size=14, multiplier=3.0, method="3sigma", strict_factor=0.8):
        self.window_size = window_size
        self.method = method
        self.multiplier = multiplier
        self.strict_factor = strict_factor

    # Nothing here -> context developed dynamiclly
    def fit(self, trein_series):
        pass

    def detect(self, test_series, trend_series):

        is_uptrend = trend_series.diff() > 0

        if self.method == '3sigma':
            # Calculate local mean and std 
            local_center = test_series.rolling(window=self.window_size).mean()
            local_spread = test_series.rolling(window=self.window_size).std()

        elif self.method == 'iqr':
            # Calculate local Q1 and Q3
            q1 = test_series.rolling(window=self.window_size).quantile(0.25)
            q3 = test_series.rolling(window=self.window_size).quantile(0.75)
            local_center_low = q1
            local_center_high = q3
            local_spread = q3 - q1

        multiplier_lower = pd.Series(self.multiplier, index=test_series.index)
        multiplier_upper = pd.Series(self.multiplier, index=test_series.index)

        multiplier_lower[is_uptrend] = self.multiplier * self.strict_factor
        multiplier_upper[~is_uptrend] = self.multiplier * self.strict_factor

        if self.method == '3sigma':
            lower_bound = local_center - (multiplier_lower * local_spread)
            upper_bound = local_center + (multiplier_upper * local_spread)
        elif self.method == 'iqr':
            lower_bound = local_center_low - (multiplier_lower * local_spread)
            upper_bound = local_center_high + (multiplier_upper * local_spread)

        anomalies_flags = (test_series < lower_bound) | (test_series > upper_bound)
        anomalies_flags = anomalies_flags.fillna(False)

        return anomalies_flags.values
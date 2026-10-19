import numpy as np
import pandas as pd

# Contextual anomaly
class newContextualAnomalyDetector:
    def __init__(self, method='3sigma', multiplier=3.0, window_size=14, strict_factor_lower=0.8, strict_factor_upper=1.2):
        self.method = method
        self.multiplier = multiplier
        self.window_size = window_size
        self.strict_factor_lower = strict_factor_lower
        self.strict_factor_upper = strict_factor_upper

    def detect(self, series, context_series):
        print("[INFO] Detecting contextual anomalies...")

        # check trend
        is_uptrend = context_series.diff().shift(1) > 0

        if self.method == '3sigma':
            roling_mean = series.rolling(window=self.window_size).mean().shift(1)
            roling_std = series.rolling(window=self.window_size).std().shift(1)
        
        elif self.method == 'iqr':
            q1 = series.rolling(window=self.window_size).quantile(0.25).shift(1)
            q3 = series.rolling(window=self.window_size).quantile(0.75).shift(1)
        
            iqr = q3 - q1

        # add multiplier to index
        multiplier_lower = pd.Series(self.multiplier, index=series.index)
        multiplier_upper = pd.Series(self.multiplier, index=series.index)

        # if trend is up, lower the lower bound and upper the upper bound
        multiplier_lower[is_uptrend] = self.multiplier * self.strict_factor_lower
        multiplier_upper[is_uptrend] = self.multiplier * self.strict_factor_upper

        # if trend is down, lower the upper bound and upper the lower bound
        multiplier_upper[~is_uptrend] = self.multiplier * self.strict_factor_lower
        multiplier_lower[~is_uptrend] = self.multiplier * self.strict_factor_upper

        if self.method == "3sigma":
            lower_bound = roling_mean - (multiplier_lower * roling_std)
            upper_bound = roling_mean + (multiplier_upper * roling_std)

        elif self.method == "iqr":
            lower_bound = q1 - (multiplier_lower * iqr)
            upper_bound = q3 + (multiplier_upper * iqr)

        anomalies_flag = (series > upper_bound) | (series < lower_bound)

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} {self.method} contextual anomalies")

        return anomalies_flag.fillna(False).values

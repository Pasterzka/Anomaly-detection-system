import numpy as np
import pandas as pd

# Detecting point anomalies
class newPointAnomalyDetector:
    def __init__(self, method='3sigma', multiplier=3.0, window_size=14):
        self.window_size = window_size
        self.method = method
        self.multiplier = multiplier

    def detect(self, series):
        print("[INFO] Detecting point anomalies...")

        if self.method == '3sigma':
            roling_mean = series.rolling(window=self.window_size).mean().shift(1)
            roling_std = series.rolling(window=self.window_size).std().shift(1)

            lower_bound = roling_mean - roling_std * self.multiplier
            upper_bound = roling_mean + roling_std * self.multiplier

        elif self.method == 'iqr':
            q1 = series.rolling(window=self.window_size).quantile(0.25).shift(1)
            q3 = series.rolling(window=self.window_size).quantile(0.75).shift(1)

            iqr = q3 - q1

            lower_bound = q1 - iqr * self.multiplier
            upper_bound = q3 + iqr * self.multiplier

        anomalies_flag = (series > upper_bound) | (series < lower_bound)

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} {self.method} point anomalies")

        return anomalies_flag.fillna(False).values
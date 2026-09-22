import numpy as np
import pandas as pd

# Collecive anomaly detector
class CollectiveAnomalyDetector:
    def __init__(self, window_size = 7, multiplier = 3.0, method='3sigma'):

        self.window_size = window_size
        self.multiplier = multiplier
        self.method = method

        # parameters for 3-sigma
        self.baseline_mean = None
        self.baseline_std = None

        # parameters for iqr
        self.q1 = None
        self.q3 = None
        self.iqr = None

    # Learning statistic for rolling aggregates on the training set
    def fit(self, train_serires):
        print(f"[INFO] Fitting CollectiveAnomalyDetector (window: {self.window_size} days)...")

        # Calculate rolling sum of returns
        rolling_sums = train_serires.rolling(window = self.window_size).sum().dropna()

        if self.method == "3sigma":
            self.baseline_mean = rolling_sums.mean()
            self.baseline_std = rolling_sums.std()

        elif self.method == "iqr":
            self.q1 = rolling_sums.quantile(0.25)
            self.q3 = rolling_sums.quantile(0.75)
            self.iqr = self.q3 - self.q1

    def detect(self, test_series): 
        print("[INFO] Detecting collective anomalies...")

        rolling_sums = test_series.rolling(window=self.window_size).sum()

        if self.method == "3sigma":
            lower_bound = self.baseline_mean - self.multiplier * self.baseline_std
            upper_bound = self.baseline_mean + self.multiplier * self.baseline_std

        elif self.method == "iqr":
            lower_bound = self.q1 - self.multiplier * self.iqr
            upper_bound = self.q3 + self.multiplier * self.iqr

        # Identify windows where the aggregate sum exceeds the statistical bounds
        anomalous_windows = (rolling_sums < lower_bound) | (rolling_sums > upper_bound)

        anomalies_flags = pd.Series(False, index=test_series.index)

        for idx in anomalous_windows[anomalous_windows].index:
            window_start = test_series.index.get_loc(idx) - self.window_size + 1
            window_end = test_series.index.get_loc(idx) + 1
            if window_start >= 0:
                anomalies_flags.iloc[window_start:window_end] = True

        return anomalies_flags.values
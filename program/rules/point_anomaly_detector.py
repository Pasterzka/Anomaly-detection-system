import numpy as np
import pandas as pd

# Point anomaly detector
class PointAnomalyDetector():
    # Inicjalizing detector - method 3sigma or igr - parameter for 3sigma usually 3.0 for iqr 1.5
    def __init__(self, method='3sigma', multiplier=3.0):

        self.method = method
        self.multiplier = multiplier

        # init parameter for 3sigma
        self.mean = None
        self.std = None

        # init parameter for iqr
        self.q1 = None
        self.q3 = None
        self.iqr = None

    # Learn on trening data
    def fit(self, train_series):

        if self.method == "3sigma":
            self.mean = train_series.mean()
            self.std = train_series.std()
        elif self.method == "iqr":
            self.q1 = train_series.quantile(0.25)
            self.q3 = train_series.quantile(0.75)
            self.iqr = self.q3 - self.q1
        else:
            print("[ERROR] Wrong method")

    # detect anomalies on test data
    def detect(self, test_series):
        if self.method == '3sigma':
            lower_bound = self.mean - self.multiplier * self.std
            upper_bound = self.mean + self.multiplier * self.std
        elif self.method == "iqr":
            lower_bound = self.q1 - self.multiplier * self.iqr
            upper_bound = self.q3 + self.multiplier * self.iqr
            
        anomalies_flags = (test_series < lower_bound) | (test_series > upper_bound)

        return anomalies_flags.values


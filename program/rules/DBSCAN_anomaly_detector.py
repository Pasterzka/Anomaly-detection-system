import shap
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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

        explanations = pd.Series("", index=series.index)
        valid_indices = series.dropna().index
        anomaly_mask = (y_pred == -1)

        for idx in valid_indices[anomaly_mask]:
            explanations.loc[idx] = f"Point classified as noise by DBSCAN."

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} DBSCAN point anomalies")

        return anomalies_flag.values, explanations.values

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

        explanations = pd.Series("", index=series.index)

        anomalies_flag = pd.Series(False, index=series.index)
        for idx in anomalies_flags[anomalies_flags == True].index:
            idx_num = series.index.get_loc(idx)
            window_start = max(0, idx_num - self.window_size + 1)
            window_end = idx_num + 1
            anomalies_flag.iloc[window_start:window_end] = True

            anomalies_flag.iloc[window_start:window_end] = True
            explanations.iloc[window_start:window_end] = "Collective anomally detected."
            explanations.loc[idx] = f"Point of detection of collective anomaly based on a noise cluster by DBSCAN"

        anomalies_num = anomalies_flags.sum()
        print(f"[SUCCESS] Detected {anomalies_num} DBSCAN colective anomalies")

        return anomalies_flag.values, explanations.values

class DBSCANContextualAnomalyDetector:
    def __init__(self, eps=0.5, min_sample=5, penalty_factor=1.2, reward_factor=0.8):
        self.eps = eps
        self.min_sample = min_sample
        self.penalty = penalty_factor
        self.reward = reward_factor

    def detect(self, series, context_series, return_series):

        # Calculate deviation
        deviation = series - context_series
        
        # Caluclate trend
        is_uptrend = context_series.diff().shift(1) > 0
        
        multiplied_deviation = deviation.copy()
        
        # Upper trend 
        multiplied_deviation[is_uptrend & (deviation<0)] *= self.penalty
        multiplied_deviation[is_uptrend & (deviation>0)] *= self.reward
        
        multiplied_deviation[~is_uptrend & (deviation>0)] *= self.penalty
        multiplied_deviation[~is_uptrend & (deviation<0)] *= self.reward
        
        # Conacat series and context series
        context = pd.concat([return_series, multiplied_deviation], axis=1).dropna()
        context.columns = ['Return', 'Trend_Deviation']
        
        X = StandardScaler().fit_transform(context.values)
    
        # Inicjalization
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_sample)
        y_pred = dbscan.fit_predict(X)
    
    
        anomalies_flag = pd.Series(False, index=return_series.index)
        anomalies_flag.loc[context.dropna().index] = (y_pred == -1)
    
        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} DBSCAN contextual anomalies")

        explanations = pd.Series("", index=return_series.index)

        if anomalies_num > 0:
            print("[INFO] Generating SHAP explanations for DBSCAN...")

            # Create k-nn module 
            nn = NearestNeighbors(n_neighbors=self.min_sample)
            nn.fit(X)

            # Distance to min_samle neighbour
            def outlier_score_func(data):
                distances, _ = nn.kneighbors(data)
                return distances[:, -1]

            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(outlier_score_func, background)

            anomalous_data = X[y_pred == -1]
            shap_values = explainer.shap_values(anomalous_data)

            anomaly_indices = context.dropna().index[y_pred == -1]

            for i, idx in enumerate(anomaly_indices):
                top_feature_idx = np.argmax(np.abs(shap_values[i]))
                top_feature_name = context.columns[top_feature_idx]
                explanations.loc[idx] = f"Key anomaly driver: {top_feature_name}. (DBSCAN Noise Point)"
    
        return anomalies_flag.values, explanations.values


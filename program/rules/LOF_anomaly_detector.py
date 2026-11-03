import shap
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

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

        explanations = pd.Series("", index=series.index)
        valid_indices = series.dropna().index
        anomaly_mask = y_pred == -1

        for idx, score in zip(valid_indices[anomaly_mask], X_scores[anomaly_mask]):
            explanations.loc[idx] = f"Point deviates significantly from its local neighborhood (LOF Score: {score:.2f})"

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} LOF point anomalies")

        return anomalies_flag.values, explanations.values, lof_scores.values

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

        explanations = pd.Series("", index=series.index)

        anomalies_flag = pd.Series(False, index=series.index)
        for idx in anomalies_flags[anomalies_flags == True].index:
            idx_num = series.index.get_loc(idx)
            window_start = max(0, idx_num - self.window_size + 1)
            window_end = idx_num + 1
            anomalies_flag.iloc[window_start:window_end] = True

            score = lof_scores.loc[idx]

            explanations.iloc[window_start:window_end] = "Collective anomally detected."

            explanations.loc[idx] = f"Point of detection of collective anomaly based on local neighborhood (LOF Score: {score:.2f})"

        anomalies_num = anomalies_flags.sum()
        print(f"[SUCCESS] Detected {anomalies_num} LOF colective anomalies")

        return anomalies_flag.values, explanations.values, lof_scores.values

class LOFContextualAnomalyDetector:
    def __init__(self, neighbours=14, contamination=0.01, penalty_factor=1.2, reward_factor=0.8):
        self.neighbours = neighbours
        self.contamination = contamination
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
        clf = LocalOutlierFactor(n_neighbors=self.neighbours, contamination=self.contamination, novelty=True)
        clf.fit(X)
        y_pred = clf.predict(X)

        X_scores = clf.negative_outlier_factor_

        anomalies_flag = pd.Series(False, index=return_series.index)
        anomalies_flag.loc[context.dropna().index] = (y_pred == -1)

        lof_scores = pd.Series(np.nan, index=return_series.index)
        lof_scores.loc[context.dropna().index] = X_scores

        explanations = pd.Series("", index=return_series.index)

        anomalies_num = anomalies_flag.sum()
        print(f"[SUCCESS] Detected {anomalies_num} LOF contextual anomalies")

        if anomalies_num > 0:
            # Build SHAP explain
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(clf.decision_function, background)
            
            # Calculate only points of anomaly
            anomalous_data = X[y_pred == -1]
            shap_values = explainer.shap_values(anomalous_data)
            
            anomaly_indices = context.dropna().index[y_pred == -1]

            # Whot has more impact on a result
            for i, idx in enumerate(anomaly_indices):
                top_feature_idx = np.argmax(np.abs(shap_values[i]))
                top_feature_name = context.columns[top_feature_idx]
                score = X_scores[y_pred == -1][i]
                
                explanations.loc[idx] = f"Key anomaly driver: {top_feature_name}. (LOF Score: {score:.2f})"

        return anomalies_flag.values,explanations.values, lof_scores.values

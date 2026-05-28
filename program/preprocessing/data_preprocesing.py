import numpy as np

# This class is responsible for preprocessing the data, including interpolation, splitting, normalization, and calculating moving windows.
class DataPreprocessor:
    def __init__(self, window_size=14):
        self.window_size = window_size

    # Method to interpolate missing data in the DataFrame
    def interpolateData(self, df):
        print("[INFO] Starting interpolation...")
        df = df.asfreq('D')
        df.interpolate(method='linear', inplace=True)
        return df

    # Method to split the data into training, validation, and test sets, and normalize them using min-max normalization
    def splitAndNormalize(self, df):
        print("[INFO] Splitting and normalizing data...")
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()

        for col in df.columns:
            train_min = df_train[col].min()
            train_max = df_train[col].max()

            df_train[col] = self.normalizeColumn(df_train[col], train_min, train_max)
            df_val[col] = self.normalizeColumn(df_val[col], train_min, train_max)
            df_test[col] = self.normalizeColumn(df_test[col], train_min, train_max)

        return df_train, df_val, df_test

    # Method to calculate moving windows for the given data array
    def normalizeColumn(self, series, min_val, max_val):
        denominator = max_val - min_val
        if denominator == 0:
            denominator = 1e-10
        return (series - min_val) / denominator

    # Method to calculate moving windows for the given data array
    def calculateMovingWindows(self, data_array):
        moving_window = []
        for i in range(len(data_array) - self.window_size + 1):
            window_data = data_array[i : i + self.window_size]
            moving_window.append(window_data)
        return np.array(moving_window)
import numpy as np

# This class is responsible for preprocessing the data, including interpolation, splitting, normalization, and calculating moving windows.
class DataPreprocessor:
    def __init__(self, window_size=14, method="3sigma", multiplier=3.0 ):
        self.window_size = window_size
        self.scalers = {}
        self.method = method
        self.multiplier = multiplier

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
        val_end = int(n * 0.7)

        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()

        # Clean the training set by removing outliers based on a 3 standard deviation threshold
        print("[INFO] Cleaning training set...")
        features_to_clean = [
            'Return', 
            #'ROC14', 
            #'RSI14', 
            #'ATR14'
        ]
        for col in features_to_clean:
            if col in df_train.columns:
                if self.method == "3sigma":
                    mean_val = df_train[col].mean()
                    std_val = df_train[col].std()

                    outlier_condition = np.abs(df_train[col] - mean_val) > (self.multiplier * std_val)
                    df_train.loc[outlier_condition, col] = mean_val

                elif self.method == "iqr":
                    q1 = df_train[col].quantile(0.25)
                    q3 = df_train[col].quantile(0.75)
                    median_val = df_train[col].quantile(0.5)
                    iqr = q3 - q1

                    lower_bound = q1 - (self.multiplier * iqr)
                    upper_bound = q3 + (self.multiplier * iqr)

                    outlier_condition = (df_train[col] < lower_bound) | (df_train[col] > upper_bound)
                    df_train.loc[outlier_condition, col] = median_val

            outlier_count = int(outlier_condition.sum())
            print(f"[SUCCESS] Found {outlier_count} outlied points.")

        for col in df.columns:
            train_min = df_train[col].min()
            train_max = df_train[col].max()

            # Save min and max 
            self.scalers[col] = {'min': train_min, 'max': train_max}

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
    
    # Method to denormalize a column using the stored min and max values
    def denormalizeColumn(self,series, col_name):
        if col_name not in self.scalers:
            raise ValueError(f"[ERROR] No saved parameters for column '{col_name}'.")

        min_val = self.scalers[col_name]['min']
        max_val = self.scalers[col_name]['max']

        return (series * (max_val - min_val)) + min_val
    
    # Method to calculate moving windows for the given data array
    def calculateMovingWindows(self, data_array):
        moving_window = []
        for i in range(len(data_array) - self.window_size + 1):
            window_data = data_array[i : i + self.window_size]
            moving_window.append(window_data)
        return np.array(moving_window)
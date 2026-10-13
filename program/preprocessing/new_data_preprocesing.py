import pandas as pd

# Data preprocessor
class newDataPreprocessor:
    def __init__(self):
        pass

    # Intrerpolating data
    def interpolateData(self, df):
        print("[INFO] Interpolating data...")
        df = df.asfreq('D')
        df.interpolate(method='linear', inplace=True)
        return df
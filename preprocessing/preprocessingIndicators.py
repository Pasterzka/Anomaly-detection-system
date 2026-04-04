import pandas as pd
import numpy as np


def preprocessingIndicators(dataFrame):

    print("[INFO] Starting indicators preprocessing...")
    window = 14

    print("[INFO] Calculating SMA...")
    dataFrame['SMA14'] = calculateSMA(dataFrame, window)
    print("[SUCCESS] SMA calculation completed.")

    return dataFrame

# window Simple Moving Average (SMA)
def calculateSMA(dataFrame, window):
    sma = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    for i in range(window - 1, len(dataFrame)):
        suma = 0
        for j in range(i - window + 1, i + 1):
            suma += close[j]
        sma[i] = suma / window

    return sma
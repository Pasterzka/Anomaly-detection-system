import pandas as pd
import numpy as np


def preprocessingIndicators(dataFrame):

    print("[INFO] Starting indicators preprocessing...")
    window = 14

    print("[INFO] Calculating SMA...")
    sma = calculateSMA(dataFrame, window)
    dataFrame['SMA14'] = sma
    print("[SUCCESS] SMA calculation completed.")

    print("[INFO] Calculating EMA...")
    ema = calculateEMA(dataFrame, window)
    dataFrame['EMA14'] = ema
    print("[SUCCESS] EMA calculation completed.")

    print("[INFO] Calculating WMA...")
    wma = calcualteWMA(dataFrame, window)
    dataFrame['WMA14'] = wma
    print("[SUCCESS] WMA calculation completed.")


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

# window Exponential Moving Average (EMA)
def calculateEMA(dataFrame, window):
    ema = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    # Initialize the first EMA value with the corresponding SMA value
    ema[window - 1] = dataFrame['SMA14'].values[window - 1]

    alpha = 2 / (window + 1)
    for i in range(window, len(dataFrame)):
        ema[i] = close[i] * alpha + ema[i - 1] * (1 - alpha)

    return ema

def calcualteWMA(dataFrame, window):
    wma = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    weights = np.arange(1, window + 1)
    print(weights)
    weightsSum = np.sum(weights)

    for i in range(window - 1, len(dataFrame)):
        suma = 0
        for j in range(i - window + 1, i + 1):
            suma += close[j] * weights[j - (i - window + 1)]
        wma[i] = suma / weightsSum
    return wma
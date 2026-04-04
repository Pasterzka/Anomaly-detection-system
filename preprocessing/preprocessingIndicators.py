import pandas as pd
import numpy as np


def preprocessingIndicators(dataFrame):

    print("[INFO] Starting indicators preprocessing...\n\n")
    window = 14

    print("[INFO] Calculating SMA...")
    sma = calculateSMA(dataFrame, window)
    dataFrame['SMA14'] = sma
    print("[SUCCESS] SMA calculation completed.\n")

    print("[INFO] Calculating EMA...")
    ema = calculateEMA(dataFrame, window)
    dataFrame['EMA14'] = ema
    print("[SUCCESS] EMA calculation completed.\n")

    print("[INFO] Calculating WMA...")
    wma = calcualteWMA(dataFrame, window)
    dataFrame['WMA14'] = wma
    print("[SUCCESS] WMA calculation completed.")

    print("[INFO] Calculating ATR...")
    atr = calculateATR(dataFrame, window)
    dataFrame['ATR14'] = atr
    print("[SUCCESS] ATR calculation completed.\n")

    print("[INFO] Calculating ROC...")
    roc = calculateROC(dataFrame, window)
    dataFrame['ROC14'] = roc
    print("[SUCCESS] ROC calculation completed.\n")

    print("[INFO] Calculating MACD...")
    macd = calculateMACD(dataFrame, window)
    dataFrame['MACD'] = macd
    print("[SUCCESS] MACD calculation completed.\n")


    return dataFrame

# window Simple Moving Average (SMA)
def calculateSMA(dataFrame, window):
    sma = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    for i in range(window - 1, len(dataFrame)):
        sum = 0
        for j in range(i - window + 1, i + 1):
            sum += close[j]
        sma[i] = sum / window

    return sma

# window Exponential Moving Average (EMA)
def calculateEMA(dataFrame, window):
    ema = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    # Initialize the first EMA value with the corresponding SMA value
    #ema[window - 1] = dataFrame['SMA14'].values[window - 1]
    sum = 0
    for j in range(0, window):
            sum += close[j]
    ema[window - 1] = sum / window

    alpha = 2 / (window + 1)
    for i in range(window, len(dataFrame)):
        ema[i] = close[i] * alpha + ema[i - 1] * (1 - alpha)

    return ema

# window Weighted Moving Average (WMA)
def calcualteWMA(dataFrame, window):
    wma = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    weights = np.arange(1, window + 1)
    weightsSum = np.sum(weights)

    for i in range(window - 1, len(dataFrame)):
        sum = 0
        for j in range(i - window + 1, i + 1):
            sum += close[j] * weights[j - (i - window + 1)]
        wma[i] = sum / weightsSum
    return wma

# day True Range (TR)
def calculateTR(dataFrame):
    tr = np.full(len(dataFrame), np.nan)
    high = dataFrame['high'].values
    low = dataFrame['low'].values
    close = dataFrame['close'].values

    tr[0] = high[0] - low[0]
    for i in range(1, len(dataFrame)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    return tr

# window Average True Range (ATR)
def calculateATR(dataFrame, window):
    atr = np.full(len(dataFrame), np.nan)
    tr = calculateTR(dataFrame)

    for i in range(window - 1, len(dataFrame)):
        sumTR = 0
        for j in range(i - window + 1, i + 1):
            sumTR += tr[j]
        atr[i] = sumTR / window

    return atr

# window Rate of Change (ROC)
def calculateROC(dataFrame, window):
    roc = np.full(len(dataFrame), np.nan)
    close = dataFrame['close'].values

    for i in range(window, len(dataFrame)):
        roc[i] = ((close[i] - close[i - window]) / close[i - window]) * 100

    return roc

# window Moving Average Convergence Divergence (MACD)
def calculateMACD(dataFrame, window):
    emaFast = calculateEMA(dataFrame, 12)
    emaSlow = calculateEMA(dataFrame, 26)
    print(emaFast)    
    print(emaSlow)
    macd = emaFast - emaSlow
    return macd
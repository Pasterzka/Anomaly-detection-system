import numpy as np
import pandas as pd

class TechnicalIndicators:

    def __init__(self, window=14):
        self.window = window

    def apply_all(self, df):
        print("[INFO] Starting indicators preprocessing...\n")
        df['SMA14'] = self.calculateSMA(df)
        df['EMA14'] = self.calculateEMA(df, self.window)
        df['WMA14'] = self.calculateWMA(df, self.window)
        df['ATR14'] = self.calculateATR(df, self.window)
        df['ROC14'] = self.calculateROC(df, self.window)
        df['MACD']  = self.calculateMACD(df, self.window)
        df['RSI14'] = self.calculateRSI(df, self.window)
        df['OBV']   = self.calculateOBV(df)
        print("[SUCCESS] All indicators calculated successfully.\n")
        return df

    # window Simple Moving Average (SMA)
    def calculateSMA(self, dataFrame):
        sma = np.full(len(dataFrame), np.nan)
        close = dataFrame['close'].values

        for i in range(self.window - 1, len(dataFrame)):
            sum = 0
            for j in range(i - self.window + 1, i + 1):
                sum += close[j]
            sma[i] = sum / self.window

        return sma
    
    
    # window Exponential Moving Average (EMA)
    def calculateEMA(self, dataFrame, window):
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
    def calculateWMA(self, dataFrame, window):
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
    def calculateTR(self, dataFrame):
        tr = np.full(len(dataFrame), np.nan)
        high = dataFrame['high'].values
        low = dataFrame['low'].values
        close = dataFrame['close'].values

        tr[0] = high[0] - low[0]
        for i in range(1, len(dataFrame)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        return tr

    # window Average True Range (ATR)
    def calculateATR(self, dataFrame, window):
        atr = np.full(len(dataFrame), np.nan)
        tr = self.calculateTR(dataFrame)

        for i in range(window - 1, len(dataFrame)):
            sumTR = 0
            for j in range(i - window + 1, i + 1):
                sumTR += tr[j]
            atr[i] = sumTR / window

        return atr

    # window Rate of Change (ROC)
    def calculateROC(self, dataFrame, window):
        roc = np.full(len(dataFrame), np.nan)
        close = dataFrame['close'].values

        for i in range(window, len(dataFrame)):
            roc[i] = ((close[i] - close[i - window]) / close[i - window]) * 100

        return roc

    # window Moving Average Convergence Divergence (MACD)
    def calculateMACD(self, dataFrame, window):
        emaFast = self.calculateEMA(dataFrame, 12)
        emaSlow = self.calculateEMA(dataFrame, 26)
        print(emaFast)    
        print(emaSlow)
        macd = emaFast - emaSlow
        return macd

    # window Relative Strength Index (RSI)
    def calculateRSI(self, dataFrame, window):
        rsi = np.full(len(dataFrame), np.nan)
        close = dataFrame['close'].values

        gains = np.zeros(len(dataFrame))
        losses = np.zeros(len(dataFrame))

        for i in range(1, len(dataFrame)):
            change = close[i] - close[i - 1]
            if change > 0:
                gains[i] = change
            else:
                losses[i] = -change

        avgGain = np.full(len(dataFrame), np.nan)
        avgLoss = np.full(len(dataFrame), np.nan)

        avgGain[window] = np.mean(gains[1:window + 1])
        avgLoss[window] = np.mean(losses[1:window + 1])

        for i in range(window + 1, len(dataFrame)):
            avgGain[i] = (avgGain[i - 1] * (window - 1) + gains[i]) / window
            avgLoss[i] = (avgLoss[i - 1] * (window - 1) + losses[i]) / window

        for i in range(window, len(dataFrame)):
            if avgLoss[i] == 0:
                rsi[i] = 100
            else:
                rs = avgGain[i] / avgLoss[i]
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    # window On-Balance Volume (OBV)
    def calculateOBV(self, dataFrame):
        obv = np.full(len(dataFrame), np.nan)
        close = dataFrame['close'].values
        volume = dataFrame['volume'].values

        obv[0] = 0
        for i in range(1, len(dataFrame)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv
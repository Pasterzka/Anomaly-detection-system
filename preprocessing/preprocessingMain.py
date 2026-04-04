import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from influxDB import influxDBConnection
from preprocessingQuery import preprocessingQuery
from preprocessingInterpolation import preprocessingInterpolation
from preprocessingIndicators import preprocessingIndicators


stock = "AAPL"


if __name__ == "__main__":

    print("[INFO] Starting preprocessing...")

    # download data frame from influxDB
    client, writeAPI, queryApi = influxDBConnection.connectToInfluxDB()
    dataFrame = preprocessingQuery(queryApi, stock)

    #print(dataFrame.head(10))
    # interpolation
    dataFrame = preprocessingInterpolation(dataFrame)

    print("[SUCCESS] Interpolation ended.")
    #print(dataFrame.head(10))

    dataFrame = preprocessingIndicators(dataFrame)
    print(dataFrame.head(50))

    dataFrame.plot(y='close', title='Close Price')
    plt.plot(dataFrame['SMA14'], label='SMA14')
    plt.plot(dataFrame['EMA14'], label='EMA14')
    plt.plot(dataFrame['WMA14'], label='WMA14')
    plt.plot(dataFrame['ATR14'], label='ATR14')
    plt.plot(dataFrame['ROC14'], label='ROC14')
    plt.plot(dataFrame['MACD'], label='MACD')
    plt.plot(dataFrame['RSI14'], label='RSI14')
    plt.plot(dataFrame['OBV'], label='OBV')

    
    plt.ylim(-5, 1000)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price with Indicators')
    plt.legend()
    plt.show()



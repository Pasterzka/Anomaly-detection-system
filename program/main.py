import sys
import os
import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt
import influxDB.influxDBDownloadAAPL as influxDB
import LSTM.lstmClass as LSTM
import preprocessing.preprocessingQuery as preprocessingQuery
import preprocessing.preprocessingIndicators as preprocessingIndicators
import preprocessing.preprocessingMovingWindow as preprocessingMovingWindow
import preprocessing.preprocessingInterpolation as preprocessingInterpolation
import preprocessing.preprocessingNormalization as preprocessingNormalization


stock = "AAPL"

if __name__ == "__main__":

    print("[INFO] Starting program...")
    client, writeAPI, queryApi = influxDB.influxDBConnection.connectToInfluxDB()

    print("[INFO] Downloading data frame from influxDB...")
    dataFrame = preprocessingQuery.preprocessingQuery(queryApi, stock)

    print("[INFO] Start preprocessing...")
    dataFrame = preprocessingInterpolation.preprocessingInterpolation(dataFrame)
    dataFrame = preprocessingIndicators.preprocessingIndicators(dataFrame)

    dataFrame.dropna(inplace=True)

    print("[INFO] Splitting and normalizing data...")
    dataFrameTrain, dataFrameVal, dataFrameTest = preprocessingNormalization.splitAndNormalizeData(dataFrame)
  
    windowSize = 14
    print("[INFO] Calculating moving window...")
    trainSequence = preprocessingMovingWindow.calculateMovingWindow(dataFrameTrain.values, windowSize)
    valSequence = preprocessingMovingWindow.calculateMovingWindow(dataFrameVal.values, windowSize)
    testSequence = preprocessingMovingWindow.calculateMovingWindow(dataFrameTest.values, windowSize)

    print("[INFO] Initializing AI Detector...")
    detector = LSTM.LSTMAnomalyDetector()

    model, threshold, anomalies = detector.trainAndDetect(trainSequence, valSequence, testSequence)

    # Get the close prices for the test set (adjusting for the window size)
    testClosePrices = dataFrameTest['close'].values[windowSize - 1:]

    # Get the indices of the anomalies
    anomalyIndices = np.where(anomalies)[0]

    # Get the corresponding close prices for the anomalies
    anomalyPrices = testClosePrices[anomalyIndices]

    plt.figure(figsize=(16, 7))
    plt.plot(testClosePrices, label=f'Cena Akcji {stock} (Znormalizowana)', color='royalblue', linewidth=1.5, zorder=1)
    plt.scatter(anomalyIndices, anomalyPrices, color='red', label='Wykryte Anomalie', s=50, zorder=5)
    plt.title(f"Detekcja Anomalii LSTM-Autoencoder dla {stock} (Zbiór Testowy)", fontsize=16, fontweight='bold')
    plt.xlabel("Oś Czasu (Kolejne Dni Zbioru Testowego)", fontsize=12)
    plt.ylabel("Znormalizowana Cena <0, 1>", fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
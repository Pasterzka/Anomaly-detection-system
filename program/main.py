import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import influxDB.influxDBDownloadAAPL as influxDB
import preprocessing.preprocessingQuery as preprocessingQuery
import preprocessing.preprocessingInterpolation as preprocessingInterpolation
import preprocessing.preprocessingIndicators as preprocessingIndicators
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

    print("[INFO] Splitting and normalizing data...")
    dataFrameTrain, dataFrameVal, dataFrameTest = preprocessingNormalization.splitAndNormalizeData(dataFrame)
    print(dataFrameTrain.tail(10))
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from influxDB import influxDBConnection
from preprocessingQuery import preprocessingQuery
from preprocessingInterpolation import preprocessingInterpolation


stock = "AAPL"


if __name__ == "__main__":

    print("[INFO] Starting preprocessing...")

    # download data frame from influxDB
    client, writeAPI, queryApi = influxDBConnection.connectToInfluxDB()
    dataFrame = preprocessingQuery(queryApi, stock)

    print(dataFrame.head(10))
    # interpolation
    dataFrame = preprocessingInterpolation(dataFrame)

    print("[SUCCESS] Interpolation ended.")
    print(dataFrame.head(10))



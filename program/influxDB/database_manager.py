import os
import sys
import pandas as pd
import yfinance as yf
import influxdb_client
from dotenv import load_dotenv
from influxdb_client.client.write_api import SYNCHRONOUS

# InfluxDBManager class to handle interactions with InfluxDB
class InfluxDBManager:

    # Initialize the InfluxDBManager with connection parameters
    def __init__(self, url="http://localhost:8086", org="PG", bucket="STOCK-db1"):
        load_dotenv()
        self.token = os.getenv("INFLUX_TOKEN")
        self.url = url
        self.org = org
        self.bucket = bucket

        self.client = influxdb_client.InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    # Method to download stock data using yfinance and store it in InfluxDB
    def downloadAndStoreData(self, stock, period="5y", interval="1d"):
        print(f"[INFO] Downloading data for {stock}...")
        ticker_obj = yf.Ticker(stock)
        data = ticker_obj.history(period=period, interval=interval)
        data.index = data.index.tz_localize(None)

        if data.empty:
            sys.exit(f"[ERROR] No data found for {stock}.")
        
        print(f"[SUCCESS] Data for {stock} downloaded. Importing to InfluxDB...")
        points = []
        for index, row in data.iterrows():
            point = (
                influxdb_client.Point(self.bucket)
                .tag("stock", stock)
                .field("open", row["Open"])
                .field("high", row["High"])
                .field("low", row["Low"])
                .field("close", row["Close"])
                .field("volume", row["Volume"])
                .time(index)
            )
            points.append(point)

        self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        print(f"[SUCCESS] Imported {len(points)} records to database.")

    # Method to fetch stock data from InfluxDB and return it as a pandas DataFrame
    def fetchStockData(self, stock):
        print(f"[INFO] Fetching data frame for {stock} from InfluxDB...")
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -5y)
        |> filter(fn: (r) => r["_measurement"] == "{self.bucket}")
        |> filter(fn: (r) => r["stock"] == "{stock}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
        '''

        df = self.query_api.query_data_frame(query)

        if df.empty:
            sys.exit("[ERROR] Data frame is empty.")

        print(f"[SUCCESS] Data frame for {stock} fetched successfully.")
        df = df.drop(columns=['result', 'table'], errors='ignore')
        df.rename(columns={'_time': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        return df
    
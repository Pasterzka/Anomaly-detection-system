import sys
import pandas as pd


bucket="STOCK-db1"
org="PG"


def preprocessingQuery(queryApi, stock):
    
    print(f"[INFO] Starting downloading data frame for {stock}...")

    query = f'''
    from(bucket: "{bucket}")
    |> range(start: -5y)
    |> filter(fn: (r) => r["_measurement"] == "{bucket}")
    |> filter(fn: (r) => r["stock"] == "{stock}")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "open", "high", "low", "close", "volume"])
    '''

    dataFrame = queryApi.query_data_frame(query)

    if dataFrame.empty:
        sys.exit("[ERROR] Data frame is empty.")

    print(f"[SUCCESS] Data frame for {stock} downloaded successfully.")

    dataFrame = dataFrame.drop(columns=['result', 'table'], errors='ignore')
    dataFrame.rename(columns={'_time': 'Date'}, inplace=True)
    dataFrame['Date'] = pd.to_datetime(dataFrame['Date']).dt.tz_localize(None)
    dataFrame.set_index('Date', inplace=True)
    dataFrame.sort_index(inplace=True)

    return dataFrame
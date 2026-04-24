import influxdb_client
import yfinance as yf
import sys
import influxDB.influxDBConnection as influxDBConnection


bucket="STOCK-db1"
org="PG"

if __name__ == "__main__":
    # connect to influxDB
    client, writeAPI, queryApi = influxDBConnection.connectToInfluxDB()

    # download data
    stock = "AAPL"
    print(f"[INFO] Downloading data for {stock}...")

    ticker_obj = yf.Ticker(stock)
    data = ticker_obj.history(period="5y", interval="1d")
    data.index = data.index.tz_localize(None)

    if data.empty:
        sys.exit(f"[ERROR] No data found for {stock}.")
    else:
        print(f"[SUCCESS] Data for {stock} downloaded successfully.")
        print(data.head(10))


    # import data 
    points = []
    print("[INFO] Start importing data...")
    for index, row in data.iterrows():
        point = (
            influxdb_client.Point(bucket)
            .tag("stock", stock)
            .field("open", row["Open"])
            .field("high", row["High"])
            .field("low", row["Low"])
            .field("close", row["Close"])
            .field("volume", row["Volume"])
            .time(index)
        )
        points.append(point)

    # write data to influxDB
    writeAPI.write(bucket=bucket, org=org, record=points)
    print(f"[SUCCES] Imported {len(points)} records to database.")

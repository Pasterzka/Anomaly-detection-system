from dotenv import load_dotenv
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import os


# load dotenv
load_dotenv()
token = os.getenv("INFLUX_TOKEN")

# connect to influxDB
url = "http://localhost:8086"
org = "PG"



def connectInfluxDB():
    client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    writeAPI = client.write_api(write_options=SYNCHRONOUS)
    queryApi = client.query_api()

    return client, writeAPI, queryApi


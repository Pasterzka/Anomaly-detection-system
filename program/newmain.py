from influxDB.database_manager import InfluxDBManager
from preprocessing.new_data_preprocesing import newDataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.new_point_anomaly_detector import newPointAnomalyDetector
from plot.new_polt_anomalies import new_plot_anomalies

def main():

    print("[INFO] Starting program...")

    STOCK = "AAPL"
    METHOD = "iqr"
    WINDOW_SIZE = 10
    MULTIPLIER = 1.5

    # Connect to influxDB
    db_manager = InfluxDBManager()
    data_frame = db_manager.fetchStockData(STOCK)

    # Data preprocesing
    preprocessor = newDataPreprocessor()
    indicators = TechnicalIndicators(window=WINDOW_SIZE)

    print("[INFO] Starting preprocesoring data...")
    data_frame = preprocessor.interpolateData(data_frame)
    data_frame = indicators.apply_all(data_frame)
    data_frame.dropna(inplace=True)

    target_feature = "close"

    # Point anomalies
    point_detector = newPointAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    point_anomalies = point_detector.detect(data_frame[target_feature])

    print("[INFO] Plotting Point Anomalies...")
    new_plot_anomalies(
        df=data_frame, 
        anomalies_array=point_anomalies, 
        target_feature=target_feature, 
        stock_name=STOCK, 
        title="Point Anomalies"
    )

if __name__ == "__main__":
    main()
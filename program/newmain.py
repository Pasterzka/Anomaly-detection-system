from influxDB.database_manager import InfluxDBManager
from preprocessing.new_data_preprocesing import newDataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.new_point_anomaly_detector import newPointAnomalyDetector
from rules.new_collective_anomaly_detector import newCollectiveAnomalyDetector
from rules.new_contextual_anomaly_detector import newContextualAnomalyDetector
from plot.new_polt_anomalies import new_plot_anomalies

def main():

    print("[INFO] Starting program...")

    STOCK = "AAPL"
    METHOD = "iqr"
    WINDOW_SIZE = 14
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

    target_feature = "Return"

    # Point anomalies
    point_detector = newPointAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    point_anomalies = point_detector.detect(data_frame[target_feature])

    # print("[INFO] Plotting Point Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=point_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Point Anomalies"
    # )

    # # Collectiv anomalies
    # collective_detector = newCollectiveAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    # collective_anomalies = collective_detector.detect(data_frame[target_feature])

    # print("[INFO] Plotting Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=collective_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Collective Anomalies"
    # )

    # Contextual anomalies
    contextual_detector = newContextualAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    contextual_anomalies = contextual_detector.detect(series=data_frame[target_feature], context_series=data_frame['SMA50'])

    print("[INFO] Plotting Contextual Anomalies...")
    new_plot_anomalies(
        df=data_frame, 
        anomalies_array=contextual_anomalies, 
        target_feature='close', 
        stock_name=STOCK, 
        title="Contextual Anomalies"
    )

if __name__ == "__main__":
    main()
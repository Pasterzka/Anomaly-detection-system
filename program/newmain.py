from influxDB.database_manager import InfluxDBManager
from preprocessing.new_data_preprocesing import newDataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.new_point_anomaly_detector import newPointAnomalyDetector
from rules.new_collective_anomaly_detector import newCollectiveAnomalyDetector
from rules.new_contextual_anomaly_detector import newContextualAnomalyDetector
from rules.LOF_anomaly_detector import LOFPointAnomalyDetector, LOFColectiveAbomalyDetector, LOFContextualAnomalyDetector
from rules.DBSCAN_anomaly_detector import DBSCANPointAnomalyDetector, DBSCANCollectiveAnomalyDetector, DBSCANContextualAnomalyDetector
from plot.new_polt_anomalies import new_plot_anomalies, lof_plot_anomalies

def main():

    print("[INFO] Starting program...")

    STOCK = "AAPL"
    METHOD = "3sigma"
    WINDOW_SIZE = 20
    MULTIPLIER = 3.0

    NEIGBOURS = 20
    CONTAMINATION = 0.05

    EPS = 0.1
    SAMPLE = 10

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
    # point_detector = newPointAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    # point_anomalies = point_detector.detect(data_frame[target_feature])

    # lof_point_detector = LOFPointAnomalyDetector(neighbours=NEIGBOURS, contamination=CONTAMINATION)
    # lof_point_anomalies, lof_point_anomaly_score = lof_point_detector.detect(data_frame[target_feature])

    # dbscan_point_detector = DBSCANPointAnomalyDetector(eps=EPS, min_sample=SAMPLE)
    # dbscan_point_anomalies = dbscan_point_detector.detect(data_frame[target_feature])

    # print("[INFO] Plotting Point Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=point_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Point Anomalies"
    # )

    # print("[INFO] Plotting LOF Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=lof_point_anomalies, 
    #     lof_scores=lof_point_anomaly_score,       
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="LOF Point Anomalies"
    # )

    # print("[INFO] Plotting DBSCAN Point Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=dbscan_point_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="DBSCAN Point Anomalies"
    # )

    # # Collectiv anomalies
    # collective_detector = newCollectiveAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    # collective_anomalies = collective_detector.detect(data_frame[target_feature])

    # lof_collective_detector = LOFColectiveAbomalyDetector(window_size=WINDOW_SIZE, neighbours=NEIGBOURS, contamination=CONTAMINATION)
    # lof_collective_anomalies, lof_collective_anomaly_score = lof_collective_detector.detect(data_frame[target_feature])

    # dbscan_collective_detector = DBSCANCollectiveAnomalyDetector(window_size=WINDOW_SIZE, eps=EPS, min_sample=SAMPLE)
    # dbscan_collective_anomalies = dbscan_collective_detector.detect(data_frame[target_feature])

    # print("[INFO] Plotting Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=collective_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Collective Anomalies"
    # )

    # print("[INFO] Plotting LOF Collective Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=lof_collective_anomalies, 
    #     lof_scores=lof_collective_anomaly_score, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="LOF Collective Anomalies"
    # )

    # print("[INFO] Plotting DBSCAN Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=dbscan_collective_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="DBSCAN Collective Anomalies"
    # )

    # Contextual anomalies
    contextual_detector = newContextualAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    contextual_anomalies = contextual_detector.detect(series=data_frame[target_feature], context_series=data_frame['SMA50'])

    lof_contextual_detector = LOFContextualAnomalyDetector(neighbours=NEIGBOURS, contamination=CONTAMINATION)
    lof_contextual_anomalies, lof_contextual_anomaly_score = lof_contextual_detector.detect(series=data_frame[target_feature], context_series=data_frame['SMA50'])

    dbscan_contextual_detector = DBSCANContextualAnomalyDetector(eps=EPS, min_sample=SAMPLE)
    dbscan_contextual_anomalies = dbscan_contextual_detector.detect(data_frame[target_feature], data_frame['SMA50'])
    

    # print("[INFO] Plotting Contextual Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=contextual_anomalies, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Contextual Anomalies"
    # )

    # print("[INFO] Plotting LOF Contextual Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame, 
    #     anomalies_array=lof_contextual_anomalies, 
    #     lof_scores=lof_contextual_anomaly_score, 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="LOF Contextual Anomalies"
    # )

    print("[INFO] Plotting DBSCAN Contextual Anomalies...")
    new_plot_anomalies(
        df=data_frame, 
        anomalies_array=dbscan_contextual_anomalies, 
        target_feature='close', 
        stock_name=STOCK, 
        title="DBSCAN Contextual Anomalies"
    )




if __name__ == "__main__":
    main()
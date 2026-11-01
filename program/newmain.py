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

    STOCK = "INTC"

    METHOD = "iqr"
    WINDOW_SIZE = 14
    MULTIPLIER = 1.5

    NEIGBOURS = 15
    CONTAMINATION = 0.1

    EPS = 0.5
    SAMPLE = 10

    # Connect to influxDB
    db_manager = InfluxDBManager()
    # db_manager.downloadAndStoreData(STOCK, period="200d")
    data_frame = db_manager.newFeatchStockData(STOCK, start_range="-200d")

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
    point_anomalies = point_detector.detect(data_frame['Return'])

    lof_point_detector = LOFPointAnomalyDetector(neighbours=NEIGBOURS, contamination=CONTAMINATION)
    lof_point_anomalies, lof_point_anomaly_score = lof_point_detector.detect(data_frame[target_feature])

    dbscan_point_detector = DBSCANPointAnomalyDetector(eps=EPS, min_sample=SAMPLE)
    dbscan_point_anomalies = dbscan_point_detector.detect(data_frame[target_feature])

    # # Collectiv anomalies
    # collective_detector = newCollectiveAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    # collective_anomalies = collective_detector.detect(data_frame[target_feature])

    # lof_collective_detector = LOFColectiveAbomalyDetector(window_size=WINDOW_SIZE, neighbours=NEIGBOURS, contamination=CONTAMINATION)
    # lof_collective_anomalies, lof_collective_anomaly_score = lof_collective_detector.detect(data_frame[target_feature])

    # dbscan_collective_detector = DBSCANCollectiveAnomalyDetector(window_size=WINDOW_SIZE, eps=EPS, min_sample=SAMPLE)
    # dbscan_collective_anomalies = dbscan_collective_detector.detect(data_frame[target_feature])

    # Contextual anomalies
    # contextual_detector = newContextualAnomalyDetector(method=METHOD, multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    # contextual_anomalies = contextual_detector.detect(series=data_frame['close'], context_series=data_frame['SMA50'], return_series=data_frame['Return'])

    # lof_contextual_detector = LOFContextualAnomalyDetector(neighbours=NEIGBOURS, contamination=CONTAMINATION)
    # lof_contextual_anomalies, lof_contextual_anomaly_score = lof_contextual_detector.detect(series=data_frame['close'], context_series=data_frame['SMA50'], return_series=data_frame['Return'])

    # dbscan_contextual_detector = DBSCANContextualAnomalyDetector(eps=EPS, min_sample=SAMPLE)
    # dbscan_contextual_anomalies = dbscan_contextual_detector.detect(data_frame['close'], data_frame['SMA50'], return_series=data_frame['Return'])


    PLOT_DAYS = 100
    df_plot = data_frame.tail(PLOT_DAYS)
    print(f"\n[INFO] Data to anlize: from {df_plot.index.min().date()} to {df_plot.index.max().date()}\n")
    
    print("[INFO] Plotting Point Anomalies...")
    new_plot_anomalies(
        df=data_frame[-PLOT_DAYS:], 
        anomalies_array=point_anomalies[-PLOT_DAYS:], 
        target_feature='close', 
        stock_name=STOCK, 
        title="Point Anomalies"
    )

    # print("[INFO] Plotting LOF Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=lof_point_anomalies[-PLOT_DAYS:], 
    #     lof_scores=lof_point_anomaly_score[-PLOT_DAYS:],       
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="LOF Point Anomalies"
    # )

    # print("[INFO] Plotting DBSCAN Point Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=dbscan_point_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="DBSCAN Point Anomalies"
    # )

    # print("[INFO] Plotting Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=collective_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Collective Anomalies"
    # )

    # print("[INFO] Plotting LOF Collective Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=lof_collective_anomalies[-PLOT_DAYS:], 
    #     lof_scores=lof_collective_anomaly_score[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="LOF Collective Anomalies"
    # )

    # print("[INFO] Plotting DBSCAN Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=dbscan_collective_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="DBSCAN Collective Anomalies"
    # )

    

    # print("[INFO] Plotting Contextual Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=contextual_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Contextual Anomalies"
    # )

    # print("[INFO] Plotting LOF Contextual Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=lof_contextual_anomaliesv[-PLOT_DAYS:], 
    #     lof_scores=lof_contextual_anomaly_score[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="LOF Contextual Anomalies"
    # )

    # print("[INFO] Plotting DBSCAN Contextual Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=dbscan_contextual_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="DBSCAN Contextual Anomalies"
    # )




if __name__ == "__main__":
    main()
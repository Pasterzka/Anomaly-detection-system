from influxDB.database_manager import InfluxDBManager
from preprocessing.new_data_preprocesing import newDataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.new_point_anomaly_detector import newPointAnomalyDetector
from rules.new_collective_anomaly_detector import newCollectiveAnomalyDetector
from rules.new_contextual_anomaly_detector import newContextualAnomalyDetector
from rules.LOF_anomaly_detector import LOFPointAnomalyDetector, LOFColectiveAbomalyDetector, LOFContextualAnomalyDetector
from rules.DBSCAN_anomaly_detector import DBSCANPointAnomalyDetector, DBSCANCollectiveAnomalyDetector, DBSCANContextualAnomalyDetector
from plot.new_polt_anomalies import new_plot_anomalies, lof_plot_anomalies, statistic_plot_anomalies
from plot.labeling_plot import export_data_for_labeling, plot_for_manual_labeling
import pandas as pd

def main():

    print("[INFO] Starting program...")

    STOCK = "INTC"

    METHOD = "iqr"
    WINDOW_SIZE = 14
    MULTIPLIER = 1.5

    NEIGBOURS = 15
    CONTAMINATION = 0.1

    EPS = 0.4
    SAMPLE = 14

    filename = "intel_labeling_dataset.csv"
    existing_df = pd.read_csv(filename, index_col='Date')

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
    iqr_point_detector = newPointAnomalyDetector(method='iqr', multiplier=1.5, window_size=7)
    iqr_point_anomalies, iqr_point_xai, iqr_upper_bound_point, iqr_lower_bound_point = iqr_point_detector.detect(data_frame['Return'])

    sigma_point_detector = newPointAnomalyDetector(method='3sigma', multiplier=2.0, window_size=WINDOW_SIZE)
    sigma_point_anomalies, sigma_point_xai, sigma_upper_bound_point, sigma_lower_bound_point  = sigma_point_detector.detect(data_frame['Return'])

    lof_point_detector = LOFPointAnomalyDetector(neighbours=NEIGBOURS, contamination=CONTAMINATION)
    lof_point_anomalies, lof_point_xai, lof_point_anomaly_score = lof_point_detector.detect(data_frame[target_feature])

    dbscan_point_detector = DBSCANPointAnomalyDetector(eps=EPS, min_sample=SAMPLE)
    dbscan_point_anomalies = dbscan_point_detector.detect(data_frame[target_feature])

    # Collectiv anomalies
    iqr_collective_detector = newCollectiveAnomalyDetector(method='iqr', multiplier=1.5, window_size=7)
    iqr_collective_anomalies, iqr_collective_xai, iqr_upper_bound_collective, iqr_lower_bound_collective = iqr_collective_detector.detect(data_frame['Return'])

    sigma_collective_detector = newCollectiveAnomalyDetector(method='3sigma', multiplier=2.0, window_size=7)
    sigma_collective_anomalies, sigma_collective_xai, sigma_upper_bound_collective, sigma_lower_bound_collective = sigma_collective_detector.detect(data_frame['Return'])

    lof_collective_detector = LOFColectiveAbomalyDetector(window_size=7, neighbours=NEIGBOURS, contamination=CONTAMINATION)
    lof_collective_anomalies, lof_collective_xai,lof_collective_anomaly_score = lof_collective_detector.detect(data_frame[target_feature])

    dbscan_collective_detector = DBSCANCollectiveAnomalyDetector(window_size=7, eps=EPS, min_sample=SAMPLE)
    dbscan_collective_anomalies = dbscan_collective_detector.detect(data_frame[target_feature])

    # Contextual anomalies
    iqr_contextual_detector = newContextualAnomalyDetector(method='iqr', multiplier=1.5, window_size=7)
    iqr_contextual_anomalies, iqr_contextual_xai, iqr_upper_bound_contextual, iqr_lower_bound_contextual = iqr_contextual_detector.detect(series=data_frame['close'], context_series=data_frame['SMA50'], return_series=data_frame['Return'])

    sigma_contextual_detector = newContextualAnomalyDetector(method='3sigma', multiplier=2.0, window_size=7)
    sigma_contextual_anomalies, sigma_contextual_xai, sigma_upper_bound_contextual, sigma_lower_bound_contextual = sigma_contextual_detector.detect(series=data_frame['close'], context_series=data_frame['SMA50'], return_series=data_frame['Return'])
    
    lof_contextual_detector = LOFContextualAnomalyDetector(neighbours=NEIGBOURS, contamination=CONTAMINATION)
    lof_contextual_anomalies, lof_contextual_xai, lof_contextual_anomaly_score = lof_contextual_detector.detect(series=data_frame['close'], context_series=data_frame['SMA50'], return_series=data_frame['Return'])

    dbscan_contextual_detector = DBSCANContextualAnomalyDetector(eps=EPS, min_sample=SAMPLE)
    dbscan_contextual_anomalies = dbscan_contextual_detector.detect(data_frame['close'], data_frame['SMA50'], return_series=data_frame['Return'])


    PLOT_DAYS = 100
    df_plot = data_frame.tail(PLOT_DAYS)
    print(f"\n[INFO] Data to anlize: from {df_plot.index.min().date()} to {df_plot.index.max().date()}\n")
    
    # print("[INFO] Plotting IQR Point Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=iqr_point_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Point Anomalies"
    # )

    # print("[INFO] Plotting 3sigma Point Anomalies...")
    # statistic_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=sigma_point_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Z-Score Point Anomalies",
    #     lower_bound=sigma_lower_bound_point[-PLOT_DAYS:],  
    #     upper_bound=sigma_upper_bound_point[-PLOT_DAYS:], 
    #     metric_feature='Return' 
    # )

    # print("[INFO] Plotting 3sigma Point Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=sigma_point_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Point Anomalies"
    # )

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

    # print("[INFO] Plotting IQR Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=iqr_collective_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Collective Anomalies"
    # )

    # print("[INFO] Plotting 3sigma Collective Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=sigma_collective_anomalies[-PLOT_DAYS:], 
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

    # print("[INFO] Plotting IQR Contextual Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=iqr_contextual_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Contextual Anomalies"
    # )

    # print("[INFO] Plotting 3sigma Contextual Anomalies...")
    # new_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=sigma_contextual_anomalies[-PLOT_DAYS:], 
    #     target_feature='close', 
    #     stock_name=STOCK, 
    #     title="Contextual Anomalies"
    # )

    # print("[INFO] Plotting LOF Contextual Anomalies...")
    # lof_plot_anomalies(
    #     df=data_frame[-PLOT_DAYS:], 
    #     anomalies_array=lof_contextual_anomalies[-PLOT_DAYS:], 
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

    existing_df['ZScore_Point'] = sigma_point_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_ZScore_Point'] = sigma_point_xai[-PLOT_DAYS:]
    existing_df['ZScore_Collective'] = sigma_collective_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_ZScore_Collective'] = sigma_collective_xai[-PLOT_DAYS:]
    existing_df['ZScore_Contextual'] = sigma_contextual_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_ZScore_Contextual'] = sigma_contextual_xai[-PLOT_DAYS:]

    existing_df['IQR_Point'] = iqr_point_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_IQR_Point'] = iqr_point_xai[-PLOT_DAYS:]
    existing_df['IQR_Collective'] = iqr_collective_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_IQR_Collective'] = iqr_collective_xai[-PLOT_DAYS:]
    existing_df['IQR_Contextual'] = iqr_contextual_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_IQR_Contextual'] = iqr_contextual_xai[-PLOT_DAYS:]

    existing_df['LOF_Point'] = lof_point_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_LOF_Point'] = lof_point_xai[-PLOT_DAYS:]
    existing_df['LOF_Collective'] = lof_collective_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_LOF_Collective'] = lof_collective_xai[-PLOT_DAYS:]
    existing_df['LOF_Contextual'] = lof_contextual_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['XAI_LOF_Contextual'] = lof_contextual_xai[-PLOT_DAYS:]

    existing_df['DBSCAN_Point'] = dbscan_point_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['DBSCAN_Collective'] = dbscan_collective_anomalies[-PLOT_DAYS:].astype(int)
    existing_df['DBSCAN_Contextual'] = dbscan_contextual_anomalies[-PLOT_DAYS:].astype(int)
    
    for col in existing_df.columns:
        if any(keyword in col for keyword in ['Point', 'Collective', 'Contextual', 'Manual_Is_Anomaly']) and not col.startswith('XAI_'):
            existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce').fillna(0).astype(int)

    existing_df.to_csv(filename, float_format='%g')




if __name__ == "__main__":
    main()
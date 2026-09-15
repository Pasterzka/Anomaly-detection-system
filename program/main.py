from influxDB.database_manager import InfluxDBManager
from preprocessing.data_preprocesing import DataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.point_anomaly_detector import PointAnomalyDetector  
from plot.plot_anomalies import plot_anomalies_with_features

import matplotlib.pyplot as plt

def main():
    print("[INFO] Starting program...")
    STOCK = "AAPL"
    
    WINDOW_SIZE = 14 
    
    db_manager = InfluxDBManager()
    data_frame = db_manager.fetchStockData(STOCK)

    # Preprocessing: interpolation and technical indicators
    preprocessor = DataPreprocessor(window_size=WINDOW_SIZE)
    indicators = TechnicalIndicators(window=WINDOW_SIZE)

    print("[INFO] Interpolating data and calculating technical indicators...")
    data_frame = preprocessor.interpolateData(data_frame)
    data_frame = indicators.apply_all(data_frame)
    data_frame.dropna(inplace=True)

    # Split and normalize the data
    print("[INFO] Splitting data into train, validation, and test sets...")
    df_train, df_val, df_test = preprocessor.splitAndNormalize(data_frame)

    # Define the target feature for point anomaly detection
    # Using 'Return' is optimal for statistical rules like 3-sigma
    target_feature = 'Return'

    print(f"[INFO] Initializing Classical Point Anomaly Detector on feature '{target_feature}'...")
    
    detector = PointAnomalyDetector(method='iqr', multiplier=1.5)
    
    detector.fit(df_train[target_feature])

    # Detect point anomalies on the unseen test set
    point_anomalies = detector.detect(df_test[target_feature])

    # Denormalize 'close' price for a realistic plot representation
    df_test['close_real'] = preprocessor.denormalizeColumn(df_test['close'], 'close')

    # Plot Point Anomalies
    print("[INFO] Plotting Classical Point Anomalies...")
    
    plot_anomalies_with_features(
        df_test=df_test, 
        anomalies_flags=point_anomalies, 
        window_size=1,  
        stock=STOCK, 
        features=[target_feature], 
        title_prefix="Classical Point Anomalies (3-Sigma)",
        style="scatter" 
    )

if __name__ == "__main__":
    main()
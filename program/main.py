from influxDB.database_manager import InfluxDBManager
from preprocessing.data_preprocesing import DataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from LSTM.lstm_anomaly_detector import LSTMAnomalyDetector
from plot.plot_anomalies import plot_anomalies_dual, plot_anomalies_with_features

import matplotlib.pyplot as plt

def main():
    print("[INFO] Starting program...")
    STOCK = "AAPL"
    WINDOW_SIZE = 14
    
    print("[INFO] Starting program...")

    db_manager = InfluxDBManager()
    data_frame = db_manager.fetchStockData(STOCK)

    # Preprocessing: interpolation, technical indicators, normalization, and moving windows
    preprocessor = DataPreprocessor(window_size=WINDOW_SIZE)
    indicators = TechnicalIndicators(window=WINDOW_SIZE)

    data_frame = preprocessor.interpolateData(data_frame)
    data_frame = indicators.apply_all(data_frame)
    data_frame.dropna(inplace=True)

    # Split and normalize the data, then calculate moving windows
    df_train, df_val, df_test = preprocessor.splitAndNormalize(data_frame)

    print("[INFO] Calculating moving windows...")

    # Select features to use for the model
    features_to_use = [
        'Return',
        'ROC14', 
        #'MACD', 
        'RSI14',
        #'ATR14',
    ]

    train_seq = preprocessor.calculateMovingWindows(df_train[features_to_use].values)
    val_seq = preprocessor.calculateMovingWindows(df_val[features_to_use].values)
    test_seq = preprocessor.calculateMovingWindows(df_test[features_to_use].values)

    print("[INFO] Initializing AI Detector...")
    num_features = train_seq.shape[2]
    
    # Initialize the LSTM Anomaly Detector with the specified window size, number of features, and training parameters
    detector = LSTMAnomalyDetector(window_size=WINDOW_SIZE, num_features=num_features, epochs=50, collective_window=5)
    detector.train(train_seq, val_seq)

    # Retrieve both point and collective anomalies
    point_anomalies, collective_anomalies = detector.detect(test_seq)

    df_test['close_real'] = preprocessor.denormalizeColumn(df_test['close'], 'close')

    # Plot Point Anomalies (as red dots)
    print("[INFO] Plotting Point Anomalies...")
    plot_anomalies_with_features(
        df_test=df_test, 
        anomalies_flags=point_anomalies, 
        window_size=WINDOW_SIZE, 
        stock=STOCK, 
        features=features_to_use, 
        title_prefix="Point Anomalies",
        style="scatter" # Używamy punktów
    )
    
    # Plot Collective Anomalies (as bold red lines)
    print("[INFO] Plotting Collective Anomalies...")
    plot_anomalies_with_features(
        df_test=df_test, 
        anomalies_flags=collective_anomalies, 
        window_size=WINDOW_SIZE, 
        stock=STOCK, 
        features=features_to_use, 
        title_prefix="Collective Anomalies",
        style="line" # Używamy segmentów linii
    )

def test():
    print("[INFO] Starting program...")
    STOCK = "AAPL"
    WINDOW_SIZE = 14
    
    print("[INFO] Starting program...")

    db_manager = InfluxDBManager()
    
    
    data_frame = db_manager.fetchStockData(STOCK)

    # Preprocessing: interpolation, technical indicators, normalization, and moving windows
    preprocessor = DataPreprocessor(window_size=WINDOW_SIZE)
    indicators = TechnicalIndicators(window=WINDOW_SIZE)

    data_frame = preprocessor.interpolateData(data_frame)
    data_frame = indicators.apply_all(data_frame)
    data_frame.dropna(inplace=True)

    print(data_frame.head())

    indictors_columns = [
        'OBV'
    ]

    data_frame[indictors_columns].plot(figsize=(16, 7), title=f"Indykatory Techniczne dla {STOCK}", fontsize=12)
    plt.show()




if __name__ == "__main__":
    main()
    #test()
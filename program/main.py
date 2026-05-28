import numpy as np
import matplotlib.pyplot as plt

from influxDB.database_manager import InfluxDBManager
from preprocessing.data_preprocesing import DataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from LSTM.lstm_anomaly_detector import LSTMAnomalyDetector


def plot_anomalies(df_test, anomalies_flags, window_size, stock):
    test_close_prices = df_test['close'].values[window_size - 1:]
    
    anomaly_indices = np.where(anomalies_flags)[0]
    anomaly_prices = test_close_prices[anomaly_indices]

    plt.figure(figsize=(16, 7))
    plt.plot(test_close_prices, label=f'Cena Akcji {stock} (Znormalizowana)', color='royalblue', linewidth=1.5, zorder=1)
    plt.scatter(anomaly_indices, anomaly_prices, color='red', label='Wykryte Anomalie', s=50, zorder=5)
    
    plt.title(f"Detekcja Anomalii LSTM-Autoencoder dla {stock} (Zbiór Testowy)", fontsize=16, fontweight='bold')
    plt.xlabel("Oś Czasu (Kolejne Dni Zbioru Testowego)", fontsize=12)
    plt.ylabel("Znormalizowana Cena <0, 1>", fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
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
    train_seq = preprocessor.calculateMovingWindows(df_train.values)
    val_seq = preprocessor.calculateMovingWindows(df_val.values)
    test_seq = preprocessor.calculateMovingWindows(df_test.values)

    print("[INFO] Initializing AI Detector...")
    num_features = train_seq.shape[2]
    
    # Initialize the LSTM Anomaly Detector with the specified window size, number of features, and training parameters
    detector = LSTMAnomalyDetector(window_size=WINDOW_SIZE, num_features=num_features, epochs=50)
    detector.train(train_seq, val_seq)
    anomalies = detector.detect(test_seq)

    # Plot the detected anomalies on the test set
    plot_anomalies(df_test, anomalies, WINDOW_SIZE, STOCK)
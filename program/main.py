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

def plot_anomalies_dual(df_test, anomalies_flags, window_size, stock):
    test_dates = df_test.index[window_size - 1:]

    test_close_prices = df_test['close_real'].values[window_size - 1:]
    test_returns = df_test['Return'].values[window_size - 1:]
    
    # Extract anomaly indices and corresponding dates, prices, and returns
    anomaly_indices = np.where(anomalies_flags)[0]
    anomaly_dates = test_dates[anomaly_indices]
    anomaly_prices = test_close_prices[anomaly_indices]
    anomaly_returns = test_returns[anomaly_indices]

    # Create a dual subplot to show both price and return with anomalies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Detekcja Anomalii LSTM dla {stock} (Zbiór Testowy)", fontsize=18, fontweight='bold')

    # Top plot: Close Price with anomalies
    ax1.plot(test_dates, test_close_prices, label=f'Cena Akcji {stock} (USD)', color='royalblue', linewidth=1.5, zorder=1)
    ax1.scatter(anomaly_dates, anomaly_prices, color='red', label='Wykryte Anomalie', s=50, zorder=5)
    ax1.set_ylabel("Cena Zamknięcia (USD)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Bottom plot: Daily Returns with anomalies
    ax2.plot(test_dates, test_returns, label='Dzienna Stopa Zwrotu', color='darkorange', linewidth=1.2, zorder=1)
    ax2.scatter(anomaly_dates, anomaly_returns, color='red', label='Wykryte Anomalie', s=50, zorder=5)
    ax2.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.8)

    ax2.set_xlabel("Data", fontsize=12)
    ax2.set_ylabel("Stopa Zwrotu", fontsize=12)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) 
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

    # Select features to use for the model
    features_to_use = [
        'Return',
        'ROC14', 
        #'MACD', 
        'RSI14',
    ]

    train_seq = preprocessor.calculateMovingWindows(df_train[features_to_use].values)
    val_seq = preprocessor.calculateMovingWindows(df_val[features_to_use].values)
    test_seq = preprocessor.calculateMovingWindows(df_test[features_to_use].values)

    print("[INFO] Initializing AI Detector...")
    num_features = train_seq.shape[2]
    
    # Initialize the LSTM Anomaly Detector with the specified window size, number of features, and training parameters
    detector = LSTMAnomalyDetector(window_size=WINDOW_SIZE, num_features=num_features, epochs=50)
    detector.train(train_seq, val_seq)
    anomalies = detector.detect(test_seq)

    # Plot the detected anomalies on the test set
    df_test['close_real'] = preprocessor.denormalizeColumn(df_test['close'], 'close')
    plot_anomalies_dual(df_test, anomalies, WINDOW_SIZE, STOCK)
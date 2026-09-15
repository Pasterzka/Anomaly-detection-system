import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

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

# Function to plot anomalies with the original price on top and used LSTM features on the bottom
def plot_anomalies_with_features(df_test, anomalies_flags, window_size, stock, features, title_prefix="Point Anomalies", style="scatter"):
    test_dates = df_test.index[window_size - 1:]
    test_close_prices = df_test['close_real'].values[window_size - 1:]
    
    # Create a dual subplot configuration
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{title_prefix} Detection for {stock} (Test Set)", fontsize=18, fontweight='bold')

    # ---------------- TOP PLOT: Close Price ----------------
    ax1.plot(test_dates, test_close_prices, label=f'{stock} Close Price (USD)', color='royalblue', linewidth=1.5, zorder=1)
    
    if style == "scatter":
        # Extract anomaly indices and corresponding dates/prices for scatter plot
        anomaly_indices = np.where(anomalies_flags)[0]
        anomaly_dates = test_dates[anomaly_indices]
        anomaly_prices = test_close_prices[anomaly_indices]
        ax1.scatter(anomaly_dates, anomaly_prices, color='red', label='Detected Anomalies', s=50, zorder=5)
        
    elif style == "line":
        # Mask normal values so only contiguous anomaly segments are plotted as lines
        anomalous_prices = ma.masked_where(~anomalies_flags, test_close_prices)
        ax1.plot(test_dates, anomalous_prices, color='red', linewidth=3.5, label='Detected Anomalies (Segments)', zorder=5)

    ax1.set_ylabel("Close Price (USD)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ---------------- BOTTOM PLOT: Features ----------------
    colors = ['darkorange', 'green', 'purple', 'brown', 'pink', 'teal']
    
    for idx, feature in enumerate(features):
        test_feature_vals = df_test[feature].values[window_size - 1:]
        c = colors[idx % len(colors)]
        
        # Plot the continuous line for the feature
        ax2.plot(test_dates, test_feature_vals, label=feature, color=c, linewidth=1.2, zorder=1, alpha=0.7)
        
        if style == "scatter":
            anomaly_indices = np.where(anomalies_flags)[0]
            anomaly_dates = test_dates[anomaly_indices]
            anomaly_feature_vals = test_feature_vals[anomaly_indices]
            ax2.scatter(anomaly_dates, anomaly_feature_vals, color='red', s=30, zorder=5)
            
        elif style == "line":
            anomalous_feature_vals = ma.masked_where(~anomalies_flags, test_feature_vals)
            # No label added here to prevent duplicating legends
            ax2.plot(test_dates, anomalous_feature_vals, color='red', linewidth=2.5, zorder=5)
            
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Normalized Feature Values", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Format the dates nicely and present the plot
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) 
    plt.show()
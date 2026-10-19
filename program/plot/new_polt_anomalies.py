import matplotlib.pyplot as plt

def new_plot_anomalies(df, anomalies_array, target_feature, stock_name, title):

    plt.figure(figsize=(12,4))

    plt.plot(df.index, df[target_feature], label=f'Price of {stock_name}', color='royalblue', linewidth=1.5, alpha=0.8)

    anomalous_dates = df.index[anomalies_array]
    anomalous_prices = df[target_feature][anomalies_array]

    plt.scatter(anomalous_dates, anomalous_prices, color='red', label='Anomaly', zorder=5, s=60)

    plt.title(f"{title} - {stock_name}", fontsize=14)
    plt.xlabel("Dates", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def lof_plot_anomalies(df, anomalies_array, lof_scores, target_feature, stock_name, title):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    # Upper diagram
    ax1.plot(df.index, df[target_feature], label=f'Price of {stock_name}', color='royalblue', linewidth=1.5, alpha=0.8)
    
    anomalous_dates = df.index[anomalies_array]
    anomalous_prices = df[target_feature][anomalies_array]

    ax1.scatter(anomalous_dates, anomalous_prices, color='red', label='Anomaly', zorder=5, s=60)
    
    ax1.set_title(f"{title}\n{stock_name}", fontsize=14)
    ax1.set_ylabel("Price", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Lower diagram
    ax2.plot(df.index, lof_scores, label='LOF Score (Negative)', color='darkorange', linewidth=1.5)
    ax2.axhline(-1, color='green', linestyle='--', alpha=0.7, label='Normal Baseline')

    ax2.scatter(anomalous_dates, lof_scores[anomalies_array], color='red', zorder=5, s=40, label='Detected Anomaly')

    ax2.set_xlabel("Dates", fontsize=12)
    ax2.set_ylabel("LOF Score", fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
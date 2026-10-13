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
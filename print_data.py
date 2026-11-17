import pandas as pd
import matplotlib.pyplot as plt

def plot_stock_data(csv_path="intel_labeling_dataset.csv"):
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    ax1.plot(df.index, df['close'], label='Cena (close)', color='royalblue', linewidth=1.5)
    if 'SMA50' in df.columns:
        ax1.plot(df.index, df['SMA50'], label='Trend (SMA50)', color='darkorange', linewidth=1.5, linestyle='--')
    ax1.set_ylabel("Cena", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    bar_colors = ['forestgreen' if r >= 0 else 'crimson' for r in df['Return']]
    ax2.bar(df.index, df['Return'], color=bar_colors, alpha=0.7, width=0.8, label='Stopa zwrotu (Return)')
    
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.6)
    ax2.set_xlabel("Data", fontsize=12)
    ax2.set_ylabel("Return", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_stock_data("intel_labeling_dataset.csv")
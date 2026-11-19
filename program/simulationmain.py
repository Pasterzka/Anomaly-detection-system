import pandas as pd
import matplotlib.pyplot as plt
from influxDB.database_manager import InfluxDBManager
from preprocessing.new_data_preprocesing import newDataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.new_point_anomaly_detector import newPointAnomalyDetector
from simulation.simulation import TradingSimulation1, TradingSimulation2


def plot_simulation_results(df, simulation_results, stock_name, anomalies_flag=None, ref_fraction="50%"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Extract trade logs for the reference allocation fraction
    trades = simulation_results[ref_fraction]['trades']
    buy_dates = [t[0] for t in trades if t[1] == 'BUY']
    buy_prices = [t[2] for t in trades if t[1] == 'BUY']
    sell_dates = [t[0] for t in trades if t[1] == 'SELL']
    sell_prices = [t[2] for t in trades if t[1] == 'SELL']

    # Top subplot
    for fraction_key, res in simulation_results.items():
        ax1.plot(df.index, res['history'], label=f"Portfolio ({fraction_key} allocation)", linewidth=1.5)

    for d in buy_dates:
        ax1.axvline(d, color='green', linestyle=':', alpha=0.35)
    for d in sell_dates:
        ax1.axvline(d, color='red', linestyle=':', alpha=0.35)

    ax1.set_title(f"Anomaly-Based Trading Simulation for {stock_name}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Bottom subplot
    ax2.plot(df.index, df['close'], color='black', label=f"Stock Price ({stock_name})", linewidth=1.2, zorder=2)

    # Highlight all detected anomalies
    if anomalies_flag is not None:
        ax2.scatter(
            df.index[anomalies_flag],
            df['close'][anomalies_flag],
            color='orange',
            s=45,
            alpha=0.7,
            zorder=3,
            label="Detected Anomaly"
        )

    # BUY markers 
    ax2.scatter(
        buy_dates,
        buy_prices,
        marker='^',
        color='green',
        s=120,
        zorder=5,
        label="BUY"
    )

    # SELL markers
    ax2.scatter(
        sell_dates,
        sell_prices,
        marker='v',
        color='red',
        s=120,
        zorder=5,
        label="SELL"
    )

    ax2.set_ylabel("Stock Price ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    print("[INFO] Starting trading simulation pipeline...")

    STOCK = "INTC"
    WINDOW_SIZE = 7
    MULTIPLIER = 1.5
    PLOT_DAYS = 100

    # Fetch stock data from InfluxDB
    db_manager = InfluxDBManager()
    data_frame = db_manager.newFeatchStockData(STOCK, start_range="-200d")

    # Data preprocessing and technical indicators
    preprocessor = newDataPreprocessor()
    indicators = TechnicalIndicators(window=WINDOW_SIZE)

    data_frame = preprocessor.interpolateData(data_frame)
    data_frame = indicators.apply_all(data_frame)
    data_frame.dropna(inplace=True)

    # Detect point anomalies
    detector = newPointAnomalyDetector(method='iqr', multiplier=MULTIPLIER, window_size=WINDOW_SIZE)
    anomalies_flag, explanations, upper_bound, lower_bound = detector.detect(data_frame['Return'])

    # Slice data to the simulation window
    df_sim = data_frame.tail(PLOT_DAYS).copy()
    anomalies_sim = anomalies_flag[-PLOT_DAYS:]

    # Run portfolio trading simulation
    simulator = TradingSimulation2(init_cash=10000, init_shares=100)
    simulation_results = simulator.run(df_sim, anomalies_sim)

    # Print simulation summary
    print("\n" + "=" * 55)
    print(f"[SUMMARY] Trading Simulation Results for {STOCK} (last {PLOT_DAYS} days)")
    print("=" * 55)

    initial_total_val = 10000.0 + (100.0 * df_sim['close'].iloc[0])
    print(f"Initial Portfolio Value: ${initial_total_val:.2f}")

    for fraction, res in simulation_results.items():
        print(f"\n--- Allocation Strategy: {fraction} ---")
        print(f"Final Portfolio Value: ${res['final_portfolio_value']:.2f}")
        print(f"Return on Investment (ROI): {res['roi_pct']:.2f}%")
        print(f"Total Trades Executed: {len(res['trades'])}")

        if len(res['trades']) > 0:
            print("  Recent trades:")
            for date, t_type, price, amount in res['trades']:
                print(f"   -> [{date.date()}] {t_type} | Price: ${price:.2f} | Shares: {amount:.2f}")

    # Plot results
    plot_simulation_results(df_sim, simulation_results, STOCK, anomalies_flag=anomalies_sim, ref_fraction="50%")


if __name__ == "__main__":
    main()
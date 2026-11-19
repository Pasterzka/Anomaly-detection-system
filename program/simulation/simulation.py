import pandas as pd
import numpy as np

class TradingSimulation1:
    def __init__(self, init_cash=10000, init_shares=100):
        self.initial_cash = init_cash
        self.initial_shares = init_shares

    def run(self, df, anomalies_flag):
        results = {}
        percentages = [0.25, 0.50, 0.75, 1.00]

        prices = df['close']
        dates = df.index

        anomaly_indices = np.where(anomalies_flag)[0]

        # For each buy-sell percentage
        for percent in percentages:
            cash = self.initial_cash
            shares = self.initial_shares

            portfolio_values = []
            trades_log = []

            last_anomaly_price = None

            for i in range(len(df)):
                current_price = prices.iloc[i]
                current_date = dates[i]

                if i in anomaly_indices:
                    if last_anomaly_price is not None:
                        # Buy Signal
                        if current_price < last_anomaly_price:
                            cash_to_spend = cash * percent
                            if cash_to_spend > 0 and current_price > 0:
                                bought_shares = cash_to_spend / current_price
                                cash -= cash_to_spend
                                shares += bought_shares
                                trades_log.append((current_date, 'BUY', current_price, bought_shares))

                        # Sell Signal
                        elif current_price > last_anomaly_price:
                            shares_to_sell = shares * percent
                            if shares_to_sell > 0:
                                cash += shares_to_sell * current_price
                                shares -= shares_to_sell
                                trades_log.append((current_date, 'SELL', current_price, shares_to_sell))
                    
                    last_anomaly_price = current_price

                total_value = cash + (shares * current_price)
                portfolio_values.append(total_value)

            initial_total_val = self.initial_cash + (self.initial_shares * prices.iloc[0])
            final_value = portfolio_values[-1]
            roi = ((final_value - initial_total_val) / initial_total_val) * 100

            results[f"{int(percent*100)}%"] = {
                "final_portfolio_value": final_value,
                "roi_pct": roi,
                "history": portfolio_values,
                "trades": trades_log
            }

        return results
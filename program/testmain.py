from influxDB.database_manager import InfluxDBManager
from preprocessing.new_data_preprocesing import newDataPreprocessor
from preprocessing.technical_indicators import TechnicalIndicators
from rules.DBSCAN_anomaly_detector import DBSCANPointAnomalyDetector
from plot.new_polt_anomalies import dbscan_plot_anomalies


def main():
    # =========================================================================
    # 1. KONFIGURACJA DANYCH I PARAMETRÓW DBSCAN POINT
    # =========================================================================
    STOCK = "INTC"
    START_RANGE = "-200d"
    INDICATOR_WINDOW = 14
    PLOT_DAYS = 100

    EPS = 0.4
    MIN_SAMPLE = 14

    # =========================================================================
    # 2. POBIERANIE I PRZETWARZANIE DANYCH
    # =========================================================================
    db_manager = InfluxDBManager()
    data_frame = db_manager.newFeatchStockData(STOCK, start_range=START_RANGE)

    preprocessor = newDataPreprocessor()
    indicators = TechnicalIndicators(window=INDICATOR_WINDOW)

    data_frame = preprocessor.interpolateData(data_frame)
    data_frame = indicators.apply_all(data_frame)
    data_frame.dropna(inplace=True)

    # =========================================================================
    # 3. DETEKCJA ANOMALII PUNKTOWYCH (DBSCAN)
    # =========================================================================
    detector = DBSCANPointAnomalyDetector(eps=EPS, min_sample=MIN_SAMPLE)
    anomalies, explanations, k_distances = detector.detect(data_frame["Return"])

    # =========================================================================
    # 4. PODSUMOWANIE I WIZUALIZACJA
    # =========================================================================
    df_plot = data_frame.tail(PLOT_DAYS)
    anomalies_plot = anomalies[-PLOT_DAYS:]
    k_dist_plot = k_distances[-PLOT_DAYS:]

    count = int(anomalies_plot.sum())
    pct = (count / len(df_plot)) * 100
    print(f"\n[SUMMARY] DBSCAN Point (eps={EPS}, min_sample={MIN_SAMPLE})")
    print(f"Window: {df_plot.index.min().date()} -> {df_plot.index.max().date()}")
    print(f"Detected: {count} / {len(df_plot)} days ({pct:.1f}%)")

    if count > 0:
        print("Anomalous dates:")
        for date, ret in zip(df_plot.index[anomalies_plot], df_plot["Return"][anomalies_plot]):
            print(f"  - {date.date()} | Return: {ret:.4f}")

    dbscan_plot_anomalies(
        df=df_plot,
        anomalies_array=anomalies_plot,
        k_distances=k_dist_plot,
        eps=EPS,
        target_feature="close",
        stock_name=STOCK,
        title=f"DBSCAN Point Anomalies (eps={EPS}, min_sample={MIN_SAMPLE})",
    )


if __name__ == "__main__":
    main()
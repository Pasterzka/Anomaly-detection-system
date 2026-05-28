# Stock Market Anomaly Detection System (LSTM-Autoencoder)

System do analizy danych giełdowych i wykrywania anomalii w ruchach cen akcji. Projekt wykorzystuje sieć neuronową typu **LSTM Autoencoder** do modelowania standardowego zachowania rynku i flagowania nietypowych odchyleń na podstawie błędu rekonstrukcji. 

System został zaprojektowany z zachowaniem zasad programowania obiektowego.

---

## Jak działa system?

1. **Pobieranie i Magazynowanie Danych:** Za pomocą biblioteki `yfinance` system pobiera historyczne dane giełdowe wybranej spółki i zapisuje je lokalnie w bazie danych szeregów czasowych **InfluxDB**.
2. **Przetwarzanie Wstępne (Preprocessing):** Dane pobrane z bazy są interpolowane (uzupełnianie braków) i normalizowane (Min-Max Scaler do zakresu <0, 1>).
3. **Analiza Techniczna:** Automatyczne wyliczanie zestawu kluczowych wskaźników giełdowych (SMA, EMA, WMA, ATR, ROC, MACD, RSI, OBV).
4. **Transformacja Okienkowa:** Konwersja liniowego ciągu danych na "okna przesuwne", co jest krytycznym wymogiem do poprawnego trenowania sieci rekurencyjnych.
5. **Detekcja Anomalii (AI):** Model **LSTM Autoencoder** uczy się standardowych wzorców na danych treningowych. Na danych testowych każda sekwencja, której błąd rekonstrukcji (MSE) przekroczy ustalony próg (95. percentyl błędów treningowych), zostaje oflagowana jako anomalia.
6. **Wizualizacja:** Prezentacja wyników na interaktywnym wykresie, z wyraźnie zaznaczonymi punktami anomalnymi na osi czasu.

---

## Struktura Projektu

Zastosowano architekturę modułową. Każda klasa odpowiada za jedną logiczną część systemu:

* `main.py` – główny skrypt orkiestrujący przepływ danych.
* `db_manager.py` – klasa `InfluxDBManager` obsługująca bazę danych i zapytania API.
* `preprocessor.py` – klasa `DataPreprocessor` odpowiedzialna za czyszczenie, podział i okna przesuwne.
* `indicators.py` – klasa `TechnicalIndicators` budująca wskaźniki giełdowe.
* `lstm_detector.py` – klasa `LSTMAnomalyDetector` zawierająca architekturę modelu, logikę trenowania i detekcji.
* `.env` – (nieśledzony w repozytorium) plik z poświadczeniami dostępu.
* `requirements.txt` – lista wymaganych pakietów Python.

---

## Wymagania Techniczne

* **Baza danych:** Uruchomiona instancja **InfluxDB**.
* **Python:** Zalecana stabilna wersja **3.11** lub **3.12**. 
  > **⚠️ Ważna uwaga dla macOS (Apple M1/M2/M3):** Najnowsza wersja 3.14 jeszcze nie ma dostoswanych bibliotek do algorytmu.

---

## Instalacja i Uruchomienie

Skopiuj pliki projektu na swój komputer i przejdź do folderu:
```bash
git clone https://github.com/Pasterzka/Anomaly-detection-system.git
cd Anomaly-detection-system
```

Aktywacja srodowiska 
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Pamiętaj aby pobrać bazę danych InfluxDB i włączyć serwis
```bash
brew services start influxdb@2
```
a następnie wygenerować w nim TOKEN



Uruchomienie programu
```bash
python3.11 program/main.py
```

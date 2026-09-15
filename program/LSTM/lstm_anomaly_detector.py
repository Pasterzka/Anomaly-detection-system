import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# This class implements an LSTM Autoencoder for anomaly detection in time series data
class LSTMAnomalyDetector:
    def __init__(self, window_size, num_features, epochs=50, batch_size=32, collective_window=5):
        self.window_size = window_size
        self.num_features = num_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.collective_window = collective_window # Window size for aggregating collective anomalies
        
        self.model = None
        self.point_threshold = None
        self.collective_threshold = None

    # Method to build the LSTM Autoencoder model
    def buildHybridAutoencoder(self):
        print("[INFO] Building LSTM Autoencoder model...")
        model = Sequential()
        model.add(Input(shape=(self.window_size, self.num_features)))

        # ENCODER
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))

        # BRIDGE
        model.add(RepeatVector(self.window_size))

        # DECODER
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu', return_sequences=True))

        # OUTPUT LAYER
        model.add(TimeDistributed(Dense(self.num_features)))

        model.compile(optimizer='adam', loss='mse')
        return model

    # Helper method to calculate the rolling mean for collective anomalies
    def calculate_rolling_mean(self, errors, window):
        weights = np.ones(window) / window
        return np.convolve(errors, weights, mode='same')
    
    # Method to train the model using the training data and validate it using the validation data
    def train(self, train_data, val_data):
        # Build the model
        self.model = self.buildHybridAutoencoder()

        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )

        print("[INFO] Starting training...")
        self.model.fit(
            x=train_data,
            y=train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(val_data, val_data),
            callbacks=[early_stopping],
            verbose=1
        )

        print("[INFO] Calculating Reconstruction Error and Threshold...")
        train_predictions = self.model.predict(train_data)

        # Calculating Mean Squared Error for each training sequence
        train_mse = np.mean(np.power(train_data[:, -1, :] - train_predictions[:, -1, :], 2), axis=1)
        self.point_threshold = np.percentile(train_mse, 90)

        train_collective_mse = self.calculate_rolling_mean(train_mse, self.collective_window)
        self.collective_threshold = np.percentile(train_collective_mse, 90)

        print(f"[INFO] Point Anomaly Threshold set to: {self.point_threshold:.5f}")
        print(f"[INFO] Collective Anomaly Threshold set to: {self.collective_threshold:.5f}")

    # Method to detect anomalies in the test data based on the trained model and calculated threshold
    def detect(self, test_data):
        if self.model is None or self.point_threshold is None or self.collective_threshold is None:
            raise ValueError("[ERROR] Model is not trained yet. Call train() before detect().")

        print("[INFO] Detecting anomalies in Test Data...")
        test_predictions = self.model.predict(test_data)

        # Calculate standard reconstruction error
        test_mse = np.mean(np.power(test_data[:, -1, :] - test_predictions[:, -1, :], 2), axis=1)

        # Calculate aggregated reconstruction error
        test_collective_mse = self.calculate_rolling_mean(test_mse, self.collective_window)

        # Flag anomailes
        point_anomalies_flags = test_mse > self.point_threshold
        collective_anomalies_flags = test_collective_mse > self.collective_threshold

        total_point = np.sum(point_anomalies_flags)
        total_collective = np.sum(collective_anomalies_flags)

        print(f"[SUCCESS] Detection finished!")
        print(f" -> Point Anomalies: {total_point}")
        print(f" -> Collective Anomalies: {total_collective}")
        
        return point_anomalies_flags, collective_anomalies_flags
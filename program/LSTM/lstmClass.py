import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

class LSTMAnomalyDetector:

    def buildHybridAutoencoder(self, windowSize, numFeatures):

        print("[INFO] Building LSTM Autoencoder model...")
        model = Sequential()

        model.add(Input(shape=(windowSize, numFeatures)))

        # ENCODER
        # 128 neurons read window
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))

        # BRIGE
        model.add(RepeatVector(windowSize))

        # DECODER
        # 64 open decoder
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu', return_sequences=True))

        # OUTPUT LAYER
        model.add(TimeDistributed(Dense(numFeatures)))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        return model
    
    def trainAndDetect(self, trainData, valData, testData):

        # download and prepare data
        windowSize = trainData.shape[1]
        numFeatures = trainData.shape[2]

        # build model
        model = self.buildHybridAutoencoder(windowSize, numFeatures)

        earlyStopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, restore_best_weights=True
        )


        history = model.fit(
            x=trainData,
            y=trainData,
            epochs=50,
            batch_size=32,
            validation_data=(valData, valData),
            callbacks=[earlyStopping],
            verbose=1
        )

        print("[INFO] Calculating Reconstruction Error and Threshold...")
        trainPredictions = model.predict(trainData)

        # Last step MSE
        trainMSE = np.mean(np.power(trainData[:, -1, :] - trainPredictions[:, -1, :], 2), axis=1)

        threshold = np.percentile(trainMSE, 95)
        print(f"[INFO] Anomaly Threshold set to: {threshold:.5f}")

        print("[INFO] Detecting anomalies in Test Data...")
        testPredictions = model.predict(testData)
        testMSE = np.mean(np.power(testData[:, -1, :] - testPredictions[:, -1, :], 2), axis=1)

        anomaliesFlags = testMSE > threshold
        totalAnomalies = np.sum(anomaliesFlags)

        print(f"[SUCCESS] Detection finished! Found {totalAnomalies} anomalies out of {len(testData)} test sequences.")

        return model, threshold, anomaliesFlags
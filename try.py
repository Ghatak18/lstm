import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

file_path = "D:\\Users\\sghat\\Desktop\\Final project\\lstm1\\signal_metrics.csv"
df = pd.read_csv(file_path)

features = ['Latency (ms)', 'Data Throughput (Mbps)', 'Signal Strength (dBm)']
df = df[features].dropna()

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.pkl")

X, y = [], []
time_steps = 10
for i in range(len(features_scaled) - time_steps):
    X.append(features_scaled[i:i+time_steps])
    y.append(features_scaled[i+time_steps][1])

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(time_steps, len(features))),
    LSTM(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

model.save("lstm_network_usage_model.h5")

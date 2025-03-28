import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
file_path = "D:/Users/sghat/Desktop/lstm/signal_metrics.csv"
df = pd.read_csv(file_path)

# Select relevant features
features = ['Latency (ms)', 'Data Throughput (Mbps)', 'Signal Strength (dBm)']
df = df[features].dropna()  # Drop missing values if any

# Normalize data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df)

# Save the scaler for later use (for test data)
joblib.dump(scaler, "scaler.pkl")

# Prepare input-output sequences for LSTM
X, y = [], []
time_steps = 10  # Window size

for i in range(len(features_scaled) - time_steps):
    X.append(features_scaled[i:i+time_steps])  # Input sequence
    y.append(features_scaled[i+time_steps][1])  # Predict 'Data Throughput'

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(time_steps, len(features))),
    LSTM(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# Save the trained model
model.save("lstm_network_usage_model.h5")
print("✅ Model saved successfully!")

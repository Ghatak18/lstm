import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load CSV
df = pd.read_csv("D:\\Users\\sghat\\Desktop\\Final project\\lstm1\\book1.csv")

# Use only numeric part
df = df[['Data Throughput (Mbps)']].dropna()

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Create sequences
X, y = [], []
time_steps = 10
for i in range(len(scaled) - time_steps):
    X.append(scaled[i:i+time_steps])
    y.append(scaled[i+time_steps])
X, y = np.array(X), np.array(y)

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential([
    Input(shape=(time_steps, 1)),
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Fit model (💡THIS creates 'history' variable)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Plot training/validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.6f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label='Actual', color='blue')
plt.plot(y_pred[:100], label='Predicted', color='red', linestyle='--')
plt.title("Actual vs Predicted Throughput (First 100)")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

scaler = joblib.load("scaler.pkl")
model = load_model("lstm_network_usage_model.h5", compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')

file_path = "D:\\Users\\sghat\\Desktop\\Final project\\lstm1\\signal_metrics.csv"
df = pd.read_csv(file_path)

features = ['Latency (ms)', 'Data Throughput (Mbps)', 'Signal Strength (dBm)']
df = df[features].dropna()

features_scaled = scaler.transform(df)

X, y = [], []
time_steps = 10
for i in range(len(features_scaled) - time_steps):
    X.append(features_scaled[i:i+time_steps])
    y.append(features_scaled[i+time_steps][1])
X, y = np.array(X), np.array(y)

split_index = int(len(X) * 0.8)
X_test = X[split_index:]
y_test = y[split_index:]

y_pred = model.predict(X_test)
actual_values = y_test.flatten()
predicted_values = y_pred.flatten()

print("Index\tPredicted\tActual\t\tError")
for i in range(len(y_test)):
    predicted = y_pred[i][0]
    actual = y_test[i]
    error = predicted - actual
    print(f"{i}\t{predicted:.6f}\t{actual:.6f}\t{error:.6f}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.6f}")

tally_df = pd.DataFrame({
    'Index': range(len(actual_values)),
    'Predicted': predicted_values,
    'Actual': actual_values,
    'Error': predicted_values - actual_values
})

print(tally_df.head(20).to_string(index=False))

group_size = 20
num_groups = math.ceil(len(y_test) / group_size)

for group in range(num_groups):
    start = group * group_size
    end = min((group + 1) * group_size, len(y_test))
    actual_values = y_test[start:end]
    predicted_values = y_pred[start:end].flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(actual_values, label='Actual', color='blue', linewidth=2)
    plt.plot(predicted_values, label='Predicted', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Data Throughput (Mbps)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

group_size = 100
actual_values = y_test[:group_size]
predicted_values = y_pred[:group_size].flatten()

axs[0].plot(actual_values, label='Actual', color='blue', linewidth=2)
axs[0].plot(predicted_values, label='Predicted', color='orange', linestyle='--', linewidth=2)
axs[0].set_ylabel('Throughput (Mbps)')
axs[0].legend()
axs[0].grid(True)

moving_preds = []
for i in range(group_size):
    if i + time_steps < len(features_scaled):
        moving_input = np.array([features_scaled[i:i + time_steps]])
        moving_pred = model.predict(moving_input, verbose=0)[0][0]
        moving_preds.append(moving_pred)
    else:
        moving_preds.append(np.nan)

axs[1].plot(actual_values, label='Actual', color='blue', linewidth=2)
axs[1].plot(moving_preds, label='Varying Predicted', color='green', linestyle='--', linewidth=2)
axs[1].set_xlabel('Sample Index')
axs[1].set_ylabel('Throughput (Mbps)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()



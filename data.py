import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Load the dataset
file_path = "D:\\Users\\sghat\\Desktop\\lstm\\signal_metrics.csv"
df = pd.read_csv(file_path)

# Select relevant columns
features = ['Latency (ms)', 'Data Throughput (Mbps)', 'Signal Strength (dBm)']
df = df[features]

# Load the SAME scaler used during training
scaler = joblib.load("scaler.pkl")  # Ensure this file exists

# Apply the same scaling to test data
df_scaled = scaler.transform(df)  # Don't use fit_transform(), only transform()

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=features)

# Define time steps for LSTM
time_steps = 10
X_test = []
for i in range(time_steps, len(df_scaled)):
    X_test.append(df_scaled.iloc[i-time_steps:i].values)

X_test = np.array(X_test)

# Save test data
joblib.dump(X_test, "test_data.pkl")

print("✅ Test data prepared and saved. Shape:", X_test.shape)  
# Expected output: (num_samples, 10, num_features)

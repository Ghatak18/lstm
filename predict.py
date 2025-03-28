import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.losses import MeanSquaredError

# Load the model and specify the loss function manually
model = load_model("lstm_network_usage_model.h5", custom_objects={"mse": MeanSquaredError()})

# Load saved model
# model = load_model("lstm_network_usage_model.h5")
print("Model loaded successfully!")

# Load dataset (for getting real-time/new test data)
file_path = "D:/Users/sghat/Desktop/lstm/signal_metrics.csv"
df = pd.read_csv(file_path)

# Preprocess new data (Same as during training)
features = df[['Latency (ms)', 'Data Throughput (Mbps)']].values
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Get last 'time_steps' data for prediction
time_steps = 10  # Keep same as training
new_data = np.array([features[-time_steps:]])  # Shape: (1, time_steps, 2)

# Predict future usage
prediction = model.predict(new_data)
print("Predicted Future Data Usage (Mbps):", prediction[0][0])

import socket
import json
import numpy as np
import joblib
from collections import deque
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError

# Load trained LSTM model & MinMaxScaler
model = load_model("lstm_network_usage_model.h5", custom_objects={"mse": MeanSquaredError()})
scaler = joblib.load("scaler.pkl")

# Server details
GENERATOR_HOST = "localhost"  # Receives data from `data_generator.py`
GENERATOR_PORT = 5000

ALLOCATOR_HOST = "localhost"  # Sends predicted data to `band_allocator.py`
ALLOCATOR_PORT = 6000

# Buffer to store the last 10 time steps
history = deque(maxlen=10)  # Stores last 10 data points

def receive_live_data():
    """Receives network data, predicts future usage, and sends it to the allocator."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((GENERATOR_HOST, GENERATOR_PORT))
            print("✅ Connected to Data Generator...")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as allocator_client:
                allocator_client.connect((ALLOCATOR_HOST, ALLOCATOR_PORT))
                print("✅ Connected to Bandwidth Allocator...")

                while True:
                    raw_data = client.recv(1024).decode("utf-8").strip()
                    
                    if not raw_data:
                        continue  # Ignore empty data
                    
                    try:
                        network_data = json.loads(raw_data)
                        print(f"📩 Received Data: {network_data}")

                        # Ensure required fields exist
                        required_keys = ["Latency (ms)", "Data Throughput (Mbps)", "Signal Strength (dBm)"]
                        if not all(key in network_data for key in required_keys):
                            print("❌ Missing Required Features! Skipping...")
                            continue

                        # Extract values
                        latency = network_data["Latency (ms)"]
                        throughput = network_data["Data Throughput (Mbps)"]
                        signal_strength = network_data["Signal Strength (dBm)"]

                        print(f"✅ Latency: {latency} ms | Throughput: {throughput} Mbps | Signal: {signal_strength} dBm")

                        # Add the new data point to history
                        history.append([latency, throughput, signal_strength])

                        # Predict only if we have 10 data points
                        if len(history) == 10:
                            predicted_usage = predict_future_usage(history)
                            print(f"🔮 Predicted Future Usage: {predicted_usage} Mbps")

                            # Send prediction to Bandwidth Allocator
                            allocator_client.sendall(json.dumps({"predicted_bandwidth": predicted_usage}).encode("utf-8"))
                            print(f"📤 Sent Prediction to Allocator: {predicted_usage}")

                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON Decode Error: {e} | Raw Data: {raw_data}")

    except ConnectionRefusedError:
        print("❌ Unable to connect. Is the Data Generator or Allocator running?")
    except Exception as e:
        print(f"⚠️ Unexpected Error: {e}")

def predict_future_usage(history_buffer):
    """Uses LSTM to predict future network usage."""
    try:
        # Convert history buffer to NumPy array
        history_array = np.array(history_buffer)  # Shape: (10, 3)

        # Normalize input
        scaled_input = scaler.transform(history_array)  # Shape: (10, 3)

        # Reshape for LSTM input format (1 sample, 10 time steps, 3 features)
        lstm_input = scaled_input.reshape((1, 10, 3))

        # Predict future bandwidth
        predicted_scaled = model.predict(lstm_input)  # Output shape: (1, 1)

        # **Modify inverse transformation for single value**
        predicted_scaled = np.array([[predicted_scaled[0][0], 0, 0]])  # Shape: (1, 3) to match scaler

        # Apply inverse transform
        predicted_values = scaler.inverse_transform(predicted_scaled)  # Now correctly shaped

        # Extract predicted bandwidth
        predicted_bandwidth = predicted_values[0][0]  # Only use the first feature (bandwidth)

        return round(predicted_bandwidth, 2)  # Round for cleaner output

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        return 0  # Default to 0 if error

if __name__ == "__main__":
    receive_live_data()

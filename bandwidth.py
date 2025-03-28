import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError

# ================================
#  ✅ Load the Trained LSTM Model
# ================================
model = load_model("lstm_network_usage_model.h5", custom_objects={"mse": MeanSquaredError()})

# ✅ Load the scaler and test data
scaler = joblib.load("scaler.pkl")
X_test = joblib.load("test_data.pkl")

# ✅ Ensure the test data has the correct shape for LSTM
X_test = X_test.reshape((X_test.shape[0], 10, -1))  # Assuming 10 time steps

# ================================
#  ✅ Predict Network Usage
# ================================
predicted_usage = model.predict(X_test).flatten()
predicted_usage = predicted_usage.reshape(-1, 1)

# ✅ Reconstruct feature array with zeros for missing features (assuming 3 features)
dummy_features = np.zeros((predicted_usage.shape[0], 3))
dummy_features[:, 1] = predicted_usage.flatten()

# ✅ Inverse transform to get real-world usage values
predicted_usage_real = scaler.inverse_transform(dummy_features)[:, 1]

# ================================
#  ✅ Bandwidth Allocation
# ================================
def allocate_bandwidth(predictions, buffer_factor=1.2):
    return predictions * buffer_factor  # Apply buffer factor to each prediction

# ✅ Compute required bandwidth per time step
required_bandwidth = allocate_bandwidth(predicted_usage_real)

# ================================
#  ✅ Bandwidth Classification
# ================================
def classify_bandwidth(bandwidth_values):
    categories = []
    for bw in bandwidth_values:
        if bw < 10:
            categories.append("Low")
        elif 10 <= bw <= 50:
            categories.append("Medium")
        else:
            categories.append("High")
    return categories

# ✅ Classify bandwidth values
bandwidth_categories = classify_bandwidth(required_bandwidth)

# ================================
#  ✅ Visual Representation (Stacked Graphs)
# ================================
time_steps = np.arange(len(required_bandwidth))  # X-axis

# ✅ Set colors for bandwidth classification
colors = ["red" if bw < 10 else "orange" if bw <= 50 else "green" for bw in required_bandwidth]

# ✅ Create subplots (Stacked Layout)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 📌 Plot 1: Allocated Bandwidth (Line Graph)
ax1.plot(time_steps, required_bandwidth, label="Allocated Bandwidth", color="blue", linewidth=2)
ax1.set_ylabel("Allocated Bandwidth (Mbps)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.legend(loc="upper left")
ax1.set_title("Allocated Bandwidth Over Time")

# 📌 Plot 2: Bandwidth Classification (Bar Chart)
ax2.bar(time_steps, required_bandwidth, color=colors, alpha=0.7, label="Bandwidth Classification")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Bandwidth (Mbps)")
ax2.set_title("Bandwidth Classification Over Time")
ax2.legend(loc="upper left")

# 🎨 Adjust layout & Show plot
plt.tight_layout()
plt.show()

# ================================
#  ✅ Print Results
# ================================
print(f"Predicted Network Usage: {predicted_usage_real}")
print(f"Recommended Bandwidth Allocation: {required_bandwidth}")
print(f"Usage Classification: {bandwidth_categories}")

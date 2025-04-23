import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load your dataset
df = pd.read_csv("D:\\Users\\sghat\\Desktop\\Final project\\lstm1\\Book1.csv")  # replace with your path

# Assume target is 'Throughput' column
target_column = 'Throughput'
features = [target_column]  # or add more columns here if multivariate

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Set sequence length
sequence_length = 20

# Create sequences
X, y = create_sequences(scaled_data, sequence_length)

# Train-test split (80-20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model architecture
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Setup EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform to get real values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate and print error
errors = y_pred_inv.flatten() - y_test_inv.flatten()
for i in range(20):  # Show 20 sample predictions
    print(f"Index {i:2d} | Predicted: {y_pred_inv[i][0]:.6f} | Actual: {y_test_inv[i][0]:.6f} | Error: {errors[i]:.6f}")

# Optional: Mean Absolute Error
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"\nMean Absolute Error on Test Data: {mae:.6f}")

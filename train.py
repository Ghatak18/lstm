import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load and prepare data
df = pd.read_csv("D:\\Users\\sghat\\Desktop\\Final project\\lstm1\\Book1.csv")
data = df[['Data Throughput (Mbps)']].values

# 1. Feature Engineering
def create_features(dataframe):
    df = dataframe.copy()
    # Rolling statistics
    df['rolling_mean_24'] = df['Data Throughput (Mbps)'].rolling(24).mean()
    df['rolling_std_6'] = df['Data Throughput (Mbps)'].rolling(6).std()
    # Difference features
    df['diff_1'] = df['Data Throughput (Mbps)'].diff(1)
    df['pct_change_4'] = df['Data Throughput (Mbps)'].pct_change(4)
    return df.dropna()

df_featured = create_features(df)
features = ['Data Throughput (Mbps)', 'rolling_mean_24', 'rolling_std_6', 'diff_1', 'pct_change_4']
data = df_featured[features].values

# 2. Data Preparation
look_back = 72  # 3-day window for hourly data
train_size = int(len(data) * 0.8)

# Split before scaling
train_data = data[:train_size]
val_data = data[train_size - look_back:]  # Maintain look_back window

# Robust scaling
scaler_x = RobustScaler()
scaler_y = RobustScaler()

train_x = scaler_x.fit_transform(train_data)
train_y = scaler_y.fit_transform(train_data[:, [0]])  # Scale only target

val_x = scaler_x.transform(val_data)
val_y = scaler_y.transform(val_data[:, [0]])

# Sequence generator with multiple features
def create_hybrid_sequences(data, target, look_back):
    X_seq, X_feat, y = [], [], []
    for i in range(look_back, len(data)):
        X_seq.append(data[i-look_back:i, 0])  # Main time series
        X_feat.append(data[i, 1:])            # Additional features
        y.append(target[i])
    return np.array(X_seq), np.array(X_feat), np.array(y)

X_train_seq, X_train_feat, y_train = create_hybrid_sequences(train_x, train_y, look_back)
X_val_seq, X_val_feat, y_val = create_hybrid_sequences(val_x, val_y, look_back)

# 3. Hybrid Model Architecture
# Time series input
seq_input = Input(shape=(look_back, 1))
x = Bidirectional(LSTM(128, return_sequences=True))(seq_input)
x = Dropout(0.4)(x)
x = Bidirectional(LSTM(64))(x)
x = Dropout(0.3)(x)

# Feature input
feat_input = Input(shape=(len(features)-1,))
y = Dense(32, activation='relu')(feat_input)
y = Dropout(0.2)(y)

# Combined model
combined = Concatenate()([x, y])
z = Dense(64, activation='relu')(combined)
z = Dense(32, activation='relu')(z)
output = Dense(1)(z)

model = Model(inputs=[seq_input, feat_input], outputs=output)

# 4. Adaptive Training Configuration
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='huber',
              metrics=['mae'])

callbacks = [
    EarlyStopping(patience=25, restore_best_weights=True),
    ModelCheckpoint('best_hybrid_model.h5', save_best_only=True)
]

# 5. Training
history = model.fit(
    [X_train_seq[..., np.newaxis], X_train_feat], y_train,
    validation_data=([X_val_seq[..., np.newaxis], X_val_feat], y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. Enhanced Evaluation
def inverse_scale(y, scaler, feature_index=0):
    dummy = np.zeros((len(y), len(features)))
    dummy[:, feature_index] = y.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index]

# Predictions
train_pred = model.predict([X_train_seq[..., np.newaxis], X_train_feat])
val_pred = model.predict([X_val_seq[..., np.newaxis], X_val_feat])

# Inverse scaling
y_train_true = inverse_scale(y_train, scaler_y)
y_train_pred = inverse_scale(train_pred, scaler_y)
y_val_true = inverse_scale(y_val, scaler_y)
y_val_pred = inverse_scale(val_pred, scaler_y)

# Metrics
print(f"Train MAE: {mean_absolute_error(y_train_true, y_train_pred):.2f} Mbps")
print(f"Validation MAE: {mean_absolute_error(y_val_true, y_val_pred):.2f} Mbps")

# Visualization
plt.figure(figsize=(15, 6))
plt.plot(y_val_true, label='Actual', alpha=0.6)
plt.plot(y_val_pred, label='Predicted', alpha=0.8)
plt.title(f"Hybrid Model Performance (MAE: {mean_absolute_error(y_val_true, y_val_pred):.2f} Mbps)")
plt.xlabel("Time Steps")
plt.ylabel("Throughput (Mbps)")
plt.legend()
plt.grid(True)
plt.show()
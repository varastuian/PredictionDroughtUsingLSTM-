import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from statsmodels.tsa.stattools import acf, pacf
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# ---------------------------
# 1. Read and Preprocess Data
# ---------------------------
data_path = 'result/40708spi.txt'
df = pd.read_csv(data_path, delimiter=' ')
series = df['spi1'].values.astype('float32')

# Plot raw series
plt.figure(figsize=(10,4))
plt.plot(series, marker='o', markersize=3)
plt.title('Raw SPI Time Series')
plt.xlabel('Time index')
plt.ylabel('SPI')
plt.show()

# ---------------------------
# 2. Wavelet Transformation (DWT)
# ---------------------------
wavelet = 'db4'  # Daubechies 4 wavelet
level = 2        # decomposition level
coeffs = pywt.wavedec(series, wavelet, level=level)
# Reconstruct the low-frequency approximation (denoised series)
approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(series))

plt.figure(figsize=(10,4))
plt.plot(series, label='Original SPI', alpha=0.6)
plt.plot(approx, label='Wavelet Approximation', linewidth=2)
plt.title('SPI with Wavelet Approximation')
plt.xlabel('Time index')
plt.legend()
plt.show()

# Use the approximation for further processing
processed_series = approx

# ---------------------------
# 3. Lag Selection using ACF/PACF
# ---------------------------
lag_max = 24
acf_vals = acf(processed_series, nlags=lag_max)
pacf_vals = pacf(processed_series, nlags=lag_max)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.stem(range(lag_max+1), acf_vals)
plt.title('ACF')
plt.xlabel('Lag')
plt.subplot(1,2,2)
plt.stem(range(lag_max+1), pacf_vals)
plt.title('PACF')
plt.xlabel('Lag')
plt.tight_layout()
plt.show()

# A simple selection: choose lags where the absolute ACF value exceeds 0.2.
selected_lags = [lag for lag in range(1, lag_max+1) if np.abs(acf_vals[lag]) > 0.2]
print("Selected lags based on ACF:", selected_lags)

# ---------------------------
# 4. Create Dataset with Selected Lags
# ---------------------------
def create_dataset(series, lags):
    """
    For each time t (starting from max(lags)), the input consists of values at times (t-lag)
    for each lag in the list and the target is the value at time t.
    """
    X, y = [], []
    max_lag = max(lags)
    for i in range(max_lag, len(series)):
        X.append([series[i - lag] for lag in lags])
        y.append(series[i])
    return np.array(X), np.array(y)

X_all, y_all = create_dataset(processed_series, selected_lags)
train_size = int(len(X_all) * 0.67)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# For LSTM, we need a 3D shape: (samples, sequence_length, features).
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ---------------------------
# 5. Define Evaluation Metrics
# ---------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def correlation_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# ---------------------------
# 6. Model 1: LSTM Model (PyTorch)
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 16
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

lstm_model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

n_epochs = 300
lstm_model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = lstm_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 50 == 0:
        lstm_model.eval()
        with torch.no_grad():
            train_pred = lstm_model(X_train_tensor)
            test_pred = lstm_model(X_test_tensor)
            train_rmse = rmse(y_train_tensor.numpy(), train_pred.numpy())
            test_rmse = rmse(y_test_tensor.numpy(), test_pred.numpy())
        print(f"LSTM Epoch {epoch+1}/{n_epochs} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        lstm_model.train()

lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(X_test_tensor).numpy().flatten()

lstm_rmse = rmse(y_test, lstm_pred)
lstm_nse = nse(y_test, lstm_pred)
lstm_cc = correlation_coef(y_test, lstm_pred)
print("\nLSTM Performance:")
print(f"RMSE: {lstm_rmse:.4f}, NSE: {lstm_nse:.4f}, Correlation Coefficient: {lstm_cc:.4f}")

# ---------------------------
# 7. Model 2: SVR (scikit-learn)
# ---------------------------
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)

svr_rmse = rmse(y_test, svr_pred)
svr_nse = nse(y_test, svr_pred)
svr_cc = correlation_coef(y_test, svr_pred)
print("\nSVR Performance:")
print(f"RMSE: {svr_rmse:.4f}, NSE: {svr_nse:.4f}, Correlation Coefficient: {svr_cc:.4f}")

# ---------------------------
# 8. Model 3: MLP (Multilayer Perceptron - MLPNN style, PyTorch)
# ---------------------------
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# For MLP, inputs are 2D: (samples, number_of_features)
X_train_mlp = torch.tensor(X_train, dtype=torch.float32)
y_train_mlp = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
X_test_mlp = torch.tensor(X_test, dtype=torch.float32)
y_test_mlp = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

mlp_model = MLPModel(input_dim=X_train.shape[1], hidden_dim=50)
criterion_mlp = nn.MSELoss()
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)

n_epochs_mlp = 300
mlp_model.train()
for epoch in range(n_epochs_mlp):
    optimizer_mlp.zero_grad()
    output = mlp_model(X_train_mlp)
    loss = criterion_mlp(output, y_train_mlp)
    loss.backward()
    optimizer_mlp.step()
    if (epoch+1) % 50 == 0:
        mlp_model.eval()
        with torch.no_grad():
            train_pred = mlp_model(X_train_mlp)
            test_pred = mlp_model(X_test_mlp)
            train_rmse = rmse(y_train_mlp.numpy(), train_pred.numpy())
            test_rmse = rmse(y_test_mlp.numpy(), test_pred.numpy())
        print(f"MLP Epoch {epoch+1}/{n_epochs_mlp} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        mlp_model.train()

mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(X_test_mlp).numpy().flatten()

mlp_rmse = rmse(y_test, mlp_pred)
mlp_nse = nse(y_test, mlp_pred)
mlp_cc = correlation_coef(y_test, mlp_pred)
print("\nMLP Performance:")
print(f"RMSE: {mlp_rmse:.4f}, NSE: {mlp_nse:.4f}, Correlation Coefficient: {mlp_cc:.4f}")

# ---------------------------
# 9. Compare Models
# ---------------------------
print("\n=== Model Comparison on Test Data ===")
print(f"LSTM: RMSE = {lstm_rmse:.4f}, NSE = {lstm_nse:.4f}, CC = {lstm_cc:.4f}")
print(f"SVR : RMSE = {svr_rmse:.4f}, NSE = {svr_nse:.4f}, CC = {svr_cc:.4f}")
print(f"MLP : RMSE = {mlp_rmse:.4f}, NSE = {mlp_nse:.4f}, CC = {mlp_cc:.4f}")

# ---------------------------
# 10. Forecast 10 Years (120 Months) into the Future
# ---------------------------
# We'll use the best model (lowest test RMSE) for forecasting.
# For iterative forecasting, we need to update the lag inputs using previous predictions.
# Here we work in the original feature space (X with selected lags).

# Select best model based on RMSE:
model_rmse = {'LSTM': lstm_rmse, 'SVR': svr_rmse, 'MLP': mlp_rmse}
best_model_name = min(model_rmse, key=model_rmse.get)
print(f"\nBest model selected for forecasting: {best_model_name}")

# We create a copy of the last available lag features from the end of our full dataset.
last_known = processed_series[-max(selected_lags):].tolist()  # a list of the last max_lag values

# Function to update lags and predict the next value
def predict_next(model_name, current_lags):
    # current_lags is a list with length equal to number of features (selected lags)
    X_input = np.array([current_lags]).astype('float32')
    if model_name == 'LSTM':
        # Reshape to (1, seq_len, 1)
        X_tensor = torch.tensor(X_input.reshape((1, X_input.shape[1], 1)), dtype=torch.float32)
        with torch.no_grad():
            pred = lstm_model(X_tensor).item()
    elif model_name == 'SVR':
        pred = svr_model.predict(X_input)[0]
    elif model_name == 'MLP':
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            pred = mlp_model(X_tensor).item()
    return pred

# Forecast next 120 months iteratively:
n_forecast = 120
forecast = []
current_lags = last_known.copy()

# Note: current_lags should have length equal to number of selected lags.
# We assume the order in current_lags matches the order in selected_lags.
for i in range(n_forecast):
    next_val = predict_next(best_model_name, current_lags)
    forecast.append(next_val)
    # Update current lags: drop the oldest and append the predicted value.
    current_lags.pop(0)
    current_lags.append(next_val)

# Plot the forecast
plt.figure(figsize=(10,5))
plt.plot(range(len(processed_series)), processed_series, label='Processed Series')
plt.plot(range(len(processed_series), len(processed_series)+n_forecast), forecast, label='10-Year Forecast', color='red', marker='o', linestyle='--')
plt.title(f'10-Year Forecast using {best_model_name}')
plt.xlabel('Time index (months)')
plt.ylabel('SPI')
plt.legend()
plt.show()

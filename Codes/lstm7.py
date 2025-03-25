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
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# ---------------------------
# 1. Read and Preprocess Data
# ---------------------------
# Adjust the file path if needed
data_path = 'result/40708spi.txt'
df = pd.read_csv(data_path, delimiter=' ')
# Assuming the file has a column 'spi1'
series = df['spi1'].values.astype('float32')

# Optionally, plot the raw series
# plt.figure(figsize=(10,4))
# plt.plot(series, marker='o', markersize=3)
# plt.title('Raw SPI Time Series')
# plt.xlabel('Time index')
# plt.ylabel('SPI')
# plt.show()

# ---------------------------
# 2. Wavelet Transformation
# ---------------------------
# We use Discrete Wavelet Transform (DWT) to decompose the signal.
# Here we decompose the series into approximation and detail coefficients.
wavelet = 'db4'  # Daubechies 4 wavelet
level = 2        # Decomposition level

coeffs = pywt.wavedec(series, wavelet, level=level)
# We will reconstruct the approximation (low frequency) component 
# as a smoother version of the series
approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(series))

# Plot the original and the approximation
# plt.figure(figsize=(10,4))
# plt.plot(series, label='Original SPI', alpha=0.6)
# plt.plot(approx, label='Wavelet Approximation', linewidth=2)
# plt.title('SPI with Wavelet Approximation')
# plt.xlabel('Time index')
# plt.legend()
# plt.show()

# For further analysis, you can use the approximation as the denoised series.
processed_series = approx

# ---------------------------
# 3. Lag Selection with ACF/PACF
# ---------------------------
# Compute ACF and PACF values
lag_max = 24
acf_vals = acf(processed_series, nlags=lag_max)
pacf_vals = pacf(processed_series, nlags=lag_max)

# Plot ACF and PACF
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

# A simple strategy: choose lags where ACF is above a threshold (e.g., 0.2)
selected_lags = [lag for lag in range(1, lag_max+1) if np.abs(acf_vals[lag]) > 0.2]
print("Selected lags based on ACF:", selected_lags)

# ---------------------------
# 4. Create Dataset Function
# ---------------------------
def create_dataset(series, lags):
    """
    Create dataset with selected lags.
    For each time t, the input consists of the values at times t-lag for each lag in lags,
    and the target is the value at time t.
    """
    X, y = [], []
    max_lag = max(lags)
    for i in range(max_lag, len(series)):
        X.append([series[i - lag] for lag in lags])
        y.append(series[i])
    return np.array(X), np.array(y)

X_all, y_all = create_dataset(processed_series, selected_lags)

# Split into train and test (e.g., 67% train)
train_size = int(len(X_all) * 0.67)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# For the LSTM, we need 3D input: (batch, seq_len, features).
# Here, treat each set of lags as a sequence.
# Reshape to (samples, sequence_length, 1)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ---------------------------
# 5. Define Evaluation Metrics
# ---------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    # Nash-Sutcliffe Efficiency
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def correlation_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# ---------------------------
# 6. Model 1: LSTM in PyTorch
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Use the last output of the sequence for prediction
        out = self.linear(out[:, -1, :])
        return out

# Convert training data to torch tensors
X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

# Create DataLoader
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 16
train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Initialize model, loss and optimizer
lstm_model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Training loop for LSTM
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
        print(f"Epoch {epoch+1}/{n_epochs} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        lstm_model.train()

# Evaluate LSTM model
lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(X_test_tensor).numpy().flatten()

lstm_rmse = rmse(y_test, lstm_pred)
lstm_nse = nse(y_test, lstm_pred)
lstm_cc = correlation_coef(y_test, lstm_pred)
print("\nLSTM Performance:")
print("RMSE: {:.4f}, NSE: {:.4f}, Correlation Coefficient: {:.4f}".format(lstm_rmse, lstm_nse, lstm_cc))

# ---------------------------
# 7. Model 2: SVR from scikit-learn
# ---------------------------
# Train an SVR model on the same lag features
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)

svr_rmse = rmse(y_test, svr_pred)
svr_nse = nse(y_test, svr_pred)
svr_cc = correlation_coef(y_test, svr_pred)
print("\nSVR Performance:")
print("RMSE: {:.4f}, NSE: {:.4f}, Correlation Coefficient: {:.4f}".format(svr_rmse, svr_nse, svr_cc))

# ---------------------------
# 8. Plot Predictions Comparison
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test, label='True SPI', marker='o')
plt.plot(lstm_pred, label='LSTM Prediction', marker='x')
plt.plot(svr_pred, label='SVR Prediction', marker='s')
plt.title('Model Predictions on Test Data')
plt.xlabel('Test sample index')
plt.ylabel('SPI')
plt.legend()
plt.show()

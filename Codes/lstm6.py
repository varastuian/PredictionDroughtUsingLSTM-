import pandas as pd
import numpy as np

# Load data and replace -99 with NaN
df = pd.read_csv('result/40708spi.txt', delimiter=' ', parse_dates=['date'])
df.replace(-99, np.nan, inplace=True)
df.interpolate(method='linear', inplace=True)  # Linear interpolation

# Select relevant SPI scales (1, 3, 6, 12 months)
spi_features = ['spi1', 'spi3', 'spi6', 'spi12']
data = df[spi_features].values.astype('float32')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF/PACF for SPI1 (repeat for other features)
plt.figure(figsize=(12, 6))
plot_acf(df['spi1'], lags=24, title='ACF for SPI1')
plot_pacf(df['spi1'], lags=24, title='PACF for SPI1')
plt.show()

import pywt

def wavelet_decomposition(signal, wavelet='db4', level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

# Apply wavelet to each SPI feature
wavelet_data = np.apply_along_axis(wavelet_decomposition, 0, data)


import torch
import torch.nn as nn

class WaveletLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # Use last timestep
        return x

# Example: Input after wavelet (adjust dimensions)
model = WaveletLSTM(input_dim=wavelet_data.shape[1])


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
train_size = int(len(wavelet_data) * 0.67)
X_train, X_test = wavelet_data[:train_size], wavelet_data[train_size:]
y_train, y_test = data[:train_size, 0], data[train_size:, 0]  # Predict SPI1

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train).unsqueeze(1)  # Add sequence dimension
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_test_t = torch.tensor(X_test).unsqueeze(1)
y_test_t = torch.tensor(y_test).unsqueeze(1)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = loss_fn(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_loss = loss_fn(test_preds, y_test_t)
        print(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Test Loss {test_loss.item():.4f}")



from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Wavelet-MLP
mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
print(f"W-MLP RMSE: {np.sqrt(mean_squared_error(y_test, mlp_preds)):.4f}")

# Wavelet-SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
svr_preds = svr.predict(X_test)
print(f"W-SVR RMSE: {np.sqrt(mean_squared_error(y_test, svr_preds)):.4f}")



def nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

# For LSTM
lstm_preds = model(X_test_t).detach().numpy().flatten()
print(f"LSTM RMSE: {np.sqrt(mean_squared_error(y_test, lstm_preds)):.4f}")
print(f"LSTM NSE: {nse(y_test, lstm_preds):.4f}")
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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# ---------------------------
# 1. Read and Preprocess Data
# ---------------------------
data_path = 'result/40708spi.txt'
df = pd.read_csv(data_path, delimiter=' ')
series = df['spi1'].values.astype('float32')

plt.figure(figsize=(10,4))
plt.plot(series, marker='o', markersize=3)
plt.title('Raw SPI Time Series')
plt.xlabel('Time index')
plt.ylabel('SPI')
plt.show()

# ---------------------------
# 2. Wavelet Transformation (DWT)
# ---------------------------
wavelet = 'db4'
level = 2
coeffs = pywt.wavedec(series, wavelet, level=level)
approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(series))

plt.figure(figsize=(10,4))
plt.plot(series, label='Original SPI', alpha=0.6)
plt.plot(approx, label='Wavelet Approximation', linewidth=2)
plt.title('SPI with Wavelet Approximation')
plt.xlabel('Time index')
plt.legend()
plt.show()

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

# Simple threshold: select lags where |ACF| > 0.2
selected_lags = [lag for lag in range(1, lag_max+1) if np.abs(acf_vals[lag]) > 0.6]
print("Selected lags based on ACF:", selected_lags)

# ---------------------------
# 4. Create Dataset with Selected Lags
# ---------------------------
def create_dataset(series, lags):
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

# For LSTM: reshape inputs to 3D (samples, sequence_length, features)
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
# 6. Model 1: LSTM (PyTorch)
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
            train_rmse_val = rmse(y_train_tensor.numpy(), train_pred.numpy())
            test_rmse_val = rmse(y_test_tensor.numpy(), test_pred.numpy())
        print(f"LSTM Epoch {epoch+1}/{n_epochs} - Train RMSE: {train_rmse_val:.4f}, Test RMSE: {test_rmse_val:.4f}")
        lstm_model.train()
lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(X_test_tensor).numpy().flatten()
lstm_rmse = rmse(y_test, lstm_pred)
lstm_nse = nse(y_test, lstm_pred)
lstm_cc = correlation_coef(y_test, lstm_pred)
print("\nLSTM Performance:")
print(f"RMSE: {lstm_rmse:.4f}, NSE: {lstm_nse:.4f}, CC: {lstm_cc:.4f}")

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
print(f"RMSE: {svr_rmse:.4f}, NSE: {svr_nse:.4f}, CC: {svr_cc:.4f}")

# ---------------------------
# 8. Model 3: MLP (PyTorch)
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
            train_rmse_val = rmse(y_train_mlp.numpy(), train_pred.numpy())
            test_rmse_val = rmse(y_test_mlp.numpy(), test_pred.numpy())
        print(f"MLP Epoch {epoch+1}/{n_epochs_mlp} - Train RMSE: {train_rmse_val:.4f}, Test RMSE: {test_rmse_val:.4f}")
        mlp_model.train()
mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(X_test_mlp).numpy().flatten()
mlp_rmse = rmse(y_test, mlp_pred)
mlp_nse = nse(y_test, mlp_pred)
mlp_cc = correlation_coef(y_test, mlp_pred)
print("\nMLP Performance:")
print(f"RMSE: {mlp_rmse:.4f}, NSE: {mlp_nse:.4f}, CC: {mlp_cc:.4f}")

# ---------------------------
# 9. Model 4: Ensemble Decision Tree (EDT) using ExtraTreesRegressor
# ---------------------------
edt_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
edt_model.fit(X_train, y_train)
edt_pred = edt_model.predict(X_test)
edt_rmse = rmse(y_test, edt_pred)
edt_nse = nse(y_test, edt_pred)
edt_cc = correlation_coef(y_test, edt_pred)
print("\nEDT (ExtraTrees) Performance:")
print(f"RMSE: {edt_rmse:.4f}, NSE: {edt_nse:.4f}, CC: {edt_cc:.4f}")

# ---------------------------
# 10. Model 5: Random Forest
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = rmse(y_test, rf_pred)
rf_nse = nse(y_test, rf_pred)
rf_cc = correlation_coef(y_test, rf_pred)
print("\nRandom Forest Performance:")
print(f"RMSE: {rf_rmse:.4f}, NSE: {rf_nse:.4f}, CC: {rf_cc:.4f}")

# ---------------------------
# 11. Compare All Models
# ---------------------------
results = {
    'LSTM': {'RMSE': lstm_rmse, 'NSE': lstm_nse, 'CC': lstm_cc},
    'SVR': {'RMSE': svr_rmse, 'NSE': svr_nse, 'CC': svr_cc},
    'MLP': {'RMSE': mlp_rmse, 'NSE': mlp_nse, 'CC': mlp_cc},
    'EDT': {'RMSE': edt_rmse, 'NSE': edt_nse, 'CC': edt_cc},
    'RF': {'RMSE': rf_rmse, 'NSE': rf_nse, 'CC': rf_cc}
}
print("\n=== Model Comparison on Test Data ===")
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE = {metrics['RMSE']:.4f}, NSE = {metrics['NSE']:.4f}, CC = {metrics['CC']:.4f}")

# ---------------------------
# 12. Taylor Diagram for Model Comparison
# ---------------------------
# The Taylor diagram compares the standard deviation and correlation of model predictions with the observations.
# Here, we compute the standard deviation for each model's prediction and plot them on a polar plot.
obs_std = np.std(y_test)
model_names = list(results.keys())
std_devs = [np.std(lstm_pred), np.std(svr_pred), np.std(mlp_pred), np.std(edt_pred), np.std(rf_pred)]
corrs = [lstm_cc, svr_cc, mlp_cc, edt_cc, rf_cc]

# Create Taylor Diagram
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, polar=True)
theta = [np.arccos(c) for c in corrs]
r = std_devs

# Plot each model as a point
for i, model_name in enumerate(model_names):
    ax.plot(theta[i], r[i], 'o', label=model_name)
    
# Plot the reference point: Observations
ax.plot(0, obs_std, 'r*', markersize=12, label='Observation')
    
# Set the limits and labels
ax.set_title('Taylor Diagram', fontsize=14)
ax.set_rlim(0, max(std_devs + [obs_std])*1.1)
ax.set_xlabel('Standard Deviation')
# Custom theta ticks to show correlation coefficient
corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
theta_ticks = [np.arccos(c) for c in corr_ticks]
ax.set_thetagrids(np.degrees(theta_ticks), labels=[str(c) for c in corr_ticks])
plt.legend(loc='upper right')
plt.show()

# ---------------------------
# 13. Forecast 10 Years (120 Months) into the Future using the Best Model
# ---------------------------
model_rmse = {name: metrics['RMSE'] for name, metrics in results.items()}
best_model_name = min(model_rmse, key=model_rmse.get)
print(f"\nBest model selected for forecasting: {best_model_name}")

# Use the best model for forecasting
# For iterative forecasting, update the lag inputs recursively.
last_known = processed_series[-max(selected_lags):].tolist()

def predict_next(model_name, current_lags):
    X_input = np.array([current_lags]).astype('float32')
    if model_name == 'LSTM':
        X_tensor = torch.tensor(X_input.reshape((1, X_input.shape[1], 1)), dtype=torch.float32)
        with torch.no_grad():
            pred = lstm_model(X_tensor).item()
    elif model_name == 'SVR':
        pred = svr_model.predict(X_input)[0]
    elif model_name == 'MLP':
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            pred = mlp_model(X_tensor).item()
    elif model_name == 'EDT':
        pred = edt_model.predict(X_input)[0]
    elif model_name == 'RF':
        pred = rf_model.predict(X_input)[0]
    return pred

n_forecast = 12
forecast = []
current_lags = last_known.copy()

for i in range(n_forecast):
    next_val = predict_next(best_model_name, current_lags)
    forecast.append(next_val)
    current_lags.pop(0)
    current_lags.append(next_val)

plt.figure(figsize=(10,5))
plt.plot(range(len(processed_series)), processed_series, label='Processed Series')
plt.plot(range(len(processed_series), len(processed_series)+n_forecast), forecast, label='10-Year Forecast', color='red', marker='o', linestyle='--')
plt.title(f'1-Year Forecast using {best_model_name}')
plt.xlabel('Time index (months)')
plt.ylabel('SPI')
plt.legend()
plt.show()

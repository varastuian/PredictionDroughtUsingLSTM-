import os
import copy
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

######################################
# 1. Read and Preprocess Data
######################################
data_path = 'result/40708spi.txt'
df = pd.read_csv(data_path, delimiter=' ')
series = df['spi1'].values.astype('float32')

plt.figure(figsize=(10,4))
plt.plot(series, marker='o', markersize=3)
plt.title('Raw SPI Time Series')
plt.xlabel('Time index')
plt.ylabel('SPI')
plt.show()

######################################
# 2. Wavelet Transformation (DWT)
######################################
wavelet = 'db4'
level = 2
coeffs = pywt.wavedec(series, wavelet, level=level)
# Approximation: a denoised/smoothed version
approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(series))
# Detail from level 1 (used for boosted methods)
detail = pywt.upcoef('d', coeffs[1], wavelet, level=level, take=len(series))

plt.figure(figsize=(10,4))
plt.plot(series, label='Original SPI', alpha=0.6)
plt.plot(approx, label='Approximation', linewidth=2)
plt.plot(detail, label='Detail (Level 1)', linestyle='--')
plt.title('SPI with Wavelet Components')
plt.xlabel('Time index')
plt.legend()
plt.show()

# For original models, we use the approximation as the processed series.
processed_series = approx

######################################
# 3. Lag Selection using ACF/PACF
######################################
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

# Select lags where |ACF| > 0.2
selected_lags = [lag for lag in range(1, lag_max+1) if np.abs(acf_vals[lag]) > 0.2]
print("Selected lags based on ACF:", selected_lags)

######################################
# 4. Create Dataset
######################################
def create_dataset(series, lags):
    X, y = [], []
    max_lag = max(lags)
    for i in range(max_lag, len(series)):
        X.append([series[i - lag] for lag in lags])
        y.append(series[i])
    return np.array(X), np.array(y)

# Dataset for "approximation-based" methods:
X_all, y_all = create_dataset(processed_series, selected_lags)
train_size = int(len(X_all) * 0.67)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# For LSTM-based models, reshape inputs to 3D: (samples, seq_length, features)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# For wavelet-boosted methods, we create a dataset by concatenating features from approx and detail.
def create_dataset_wavelet(approx_series, detail_series, lags):
    X, y = [], []
    max_lag = max(lags)
    for i in range(max_lag, len(approx_series)):
        feat_approx = [approx_series[i - lag] for lag in lags]
        feat_detail = [detail_series[i - lag] for lag in lags]
        X.append(feat_approx + feat_detail)  # concatenate to get 2*len(lags) features
        y.append(approx_series[i])  # target still from approx
    return np.array(X), np.array(y)

X_all_wave, y_all_wave = create_dataset_wavelet(approx, detail, selected_lags)
X_train_wave, X_test_wave = X_all_wave[:train_size], X_all_wave[train_size:]
y_train_wave, y_test_wave = y_all_wave[:train_size], y_all_wave[train_size:]
# For WBi-LSTM, reshape to (samples, seq_length, 2), where seq_length = len(selected_lags)
X_train_wave_lstm = X_train_wave.reshape((X_train_wave.shape[0], len(selected_lags), 2))
X_test_wave_lstm = X_test_wave.reshape((X_test_wave.shape[0], len(selected_lags), 2))

######################################
# 5. Define Evaluation Metrics
######################################
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def correlation_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

######################################
# 6. Model 1: LSTM (Using approx dataset)
######################################
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
            print(f"LSTM Epoch {epoch+1}/{n_epochs} - Train RMSE: {rmse(y_train_tensor.numpy(), train_pred.numpy()):.4f}, Test RMSE: {rmse(y_test_tensor.numpy(), test_pred.numpy()):.4f}")
        lstm_model.train()
lstm_model.eval()
with torch.no_grad():
    lstm_pred = lstm_model(X_test_tensor).numpy().flatten()
lstm_rmse = rmse(y_test, lstm_pred)
lstm_nse = nse(y_test, lstm_pred)
lstm_cc = correlation_coef(y_test, lstm_pred)
print("\nLSTM Performance:")
print(f"RMSE: {lstm_rmse:.4f}, NSE: {lstm_nse:.4f}, CC: {lstm_cc:.4f}")

######################################
# 7. Model 2: SVR (Using approx dataset)
######################################
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_rmse = rmse(y_test, svr_pred)
svr_nse = nse(y_test, svr_pred)
svr_cc = correlation_coef(y_test, svr_pred)
print("\nSVR Performance:")
print(f"RMSE: {svr_rmse:.4f}, NSE: {svr_nse:.4f}, CC: {svr_cc:.4f}")

######################################
# 8. Model 3: MLP (Using approx dataset)
######################################
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
            print(f"MLP Epoch {epoch+1}/{n_epochs_mlp} - Train RMSE: {rmse(y_train_mlp.numpy(), train_pred.numpy()):.4f}, Test RMSE: {rmse(y_test_mlp.numpy(), test_pred.numpy()):.4f}")
        mlp_model.train()
mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(X_test_mlp).numpy().flatten()
mlp_rmse = rmse(y_test, mlp_pred)
mlp_nse = nse(y_test, mlp_pred)
mlp_cc = correlation_coef(y_test, mlp_pred)
print("\nMLP Performance:")
print(f"RMSE: {mlp_rmse:.4f}, NSE: {mlp_nse:.4f}, CC: {mlp_cc:.4f}")

######################################
# 9. Model 4: EDT (ExtraTrees) (Using approx dataset)
######################################
edt_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
edt_model.fit(X_train, y_train)
edt_pred = edt_model.predict(X_test)
edt_rmse = rmse(y_test, edt_pred)
edt_nse = nse(y_test, edt_pred)
edt_cc = correlation_coef(y_test, edt_pred)
print("\nEDT (ExtraTrees) Performance:")
print(f"RMSE: {edt_rmse:.4f}, NSE: {edt_nse:.4f}, CC: {edt_cc:.4f}")

######################################
# 10. Model 5: Random Forest (Using approx dataset)
######################################
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = rmse(y_test, rf_pred)
rf_nse = nse(y_test, rf_pred)
rf_cc = correlation_coef(y_test, rf_pred)
print("\nRandom Forest Performance:")
print(f"RMSE: {rf_rmse:.4f}, NSE: {rf_nse:.4f}, CC: {rf_cc:.4f}")

######################################
# 11. Model 6: Bootstrapped Random Forest (BRF)
######################################
n_boot = 30
brf_preds = np.zeros_like(y_test, dtype=float)
for i in range(n_boot):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train[idx]
    y_boot = y_train[idx]
    brf_model = RandomForestRegressor(n_estimators=100, random_state=42 + i)
    brf_model.fit(X_boot, y_boot)
    brf_preds += brf_model.predict(X_test)
brf_pred = brf_preds / n_boot
brf_rmse = rmse(y_test, brf_pred)
brf_nse = nse(y_test, brf_pred)
brf_cc = correlation_coef(y_test, brf_pred)
print("\nBootstrapped Random Forest (BRF) Performance:")
print(f"RMSE: {brf_rmse:.4f}, NSE: {brf_nse:.4f}, CC: {brf_cc:.4f}")

######################################
# 12. Model 7: Bi-directional LSTM (Bi-LSTM) (Using approx dataset)
######################################
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

X_train_tensor_bi = torch.tensor(X_train_lstm, dtype=torch.float32)
X_test_tensor_bi = torch.tensor(X_test_lstm, dtype=torch.float32)
bi_lstm_model = BiLSTMModel(input_size=1, hidden_size=50, num_layers=1)
criterion_bi = nn.MSELoss()
optimizer_bi = optim.Adam(bi_lstm_model.parameters(), lr=0.001)
n_epochs_bi = 300
bi_lstm_model.train()
for epoch in range(n_epochs_bi):
    for X_batch, y_batch in train_loader:
        optimizer_bi.zero_grad()
        output = bi_lstm_model(X_batch)
        loss = criterion_bi(output, y_batch)
        loss.backward()
        optimizer_bi.step()
    if (epoch+1) % 50 == 0:
        bi_lstm_model.eval()
        with torch.no_grad():
            train_pred = bi_lstm_model(X_train_tensor_bi)
            test_pred = bi_lstm_model(X_test_tensor_bi)
            print(f"Bi-LSTM Epoch {epoch+1}/{n_epochs_bi} - Train RMSE: {rmse(y_train_tensor.numpy(), train_pred.numpy()):.4f}, Test RMSE: {rmse(y_test_tensor.numpy(), test_pred.numpy()):.4f}")
        bi_lstm_model.train()
bi_lstm_model.eval()
with torch.no_grad():
    bilstm_pred = bi_lstm_model(X_test_tensor_bi).numpy().flatten()
bilstm_rmse = rmse(y_test, bilstm_pred)
bilstm_nse = nse(y_test, bilstm_pred)
bilstm_cc = correlation_coef(y_test, bilstm_pred)
print("\nBi-LSTM Performance:")
print(f"RMSE: {bilstm_rmse:.4f}, NSE: {bilstm_nse:.4f}, CC: {bilstm_cc:.4f}")

######################################
# 13. Model 8: Wavelet-Boosted Random Forest (WBRF)
######################################
wbrf_model = RandomForestRegressor(n_estimators=100, random_state=42)
wbrf_model.fit(X_train_wave, y_train_wave)
wbrf_pred = wbrf_model.predict(X_test_wave)
wbrf_rmse = rmse(y_test_wave, wbrf_pred)
wbrf_nse = nse(y_test_wave, wbrf_pred)
wbrf_cc = correlation_coef(y_test_wave, wbrf_pred)
print("\nWavelet-Boosted Random Forest (WBRF) Performance:")
print(f"RMSE: {wbrf_rmse:.4f}, NSE: {wbrf_nse:.4f}, CC: {wbrf_cc:.4f}")

######################################
# 14. Model 9: Wavelet-Boosted Bi-LSTM (WBi-LSTM)
######################################
class WBiLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1):
        super(WBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

X_train_tensor_wbi = torch.tensor(X_train_wave_lstm, dtype=torch.float32)
X_test_tensor_wbi = torch.tensor(X_test_wave_lstm, dtype=torch.float32)
wbilstm_model = WBiLSTMModel(input_size=2, hidden_size=50, num_layers=1)
criterion_wbi = nn.MSELoss()
optimizer_wbi = optim.Adam(wbilstm_model.parameters(), lr=0.001)
n_epochs_wbi = 300
wbilstm_model.train()
for epoch in range(n_epochs_wbi):
    optimizer_wbi.zero_grad()
    output = wbilstm_model(X_train_tensor_wbi)
    loss = criterion_wbi(output, torch.tensor(y_train_wave.reshape(-1,1), dtype=torch.float32))
    loss.backward()
    optimizer_wbi.step()
    if (epoch+1) % 50 == 0:
        wbilstm_model.eval()
        with torch.no_grad():
            train_pred = wbilstm_model(X_train_tensor_wbi)
            test_pred = wbilstm_model(X_test_tensor_wbi)
            print(f"WBi-LSTM Epoch {epoch+1}/{n_epochs_wbi} - Train RMSE: {rmse(torch.tensor(y_train_wave.reshape(-1,1)).numpy(), train_pred.numpy()):.4f}, Test RMSE: {rmse(torch.tensor(y_test_wave.reshape(-1,1)).numpy(), test_pred.numpy()):.4f}")
        wbilstm_model.train()
wbilstm_model.eval()
with torch.no_grad():
    wbilstm_pred = wbilstm_model(X_test_tensor_wbi).numpy().flatten()
wbilstm_rmse = rmse(y_test_wave, wbilstm_pred)
wbilstm_nse = nse(y_test_wave, wbilstm_pred)
wbilstm_cc = correlation_coef(y_test_wave, wbilstm_pred)
print("\nWavelet-Boosted Bi-LSTM (WBi-LSTM) Performance:")
print(f"RMSE: {wbilstm_rmse:.4f}, NSE: {wbilstm_nse:.4f}, CC: {wbilstm_cc:.4f}")

######################################
# 15. Compare All Models
######################################
results = {
    'LSTM': {'RMSE': lstm_rmse, 'NSE': lstm_nse, 'CC': lstm_cc},
    'SVR': {'RMSE': svr_rmse, 'NSE': svr_nse, 'CC': svr_cc},
    'MLP': {'RMSE': mlp_rmse, 'NSE': mlp_nse, 'CC': mlp_cc},
    'EDT': {'RMSE': edt_rmse, 'NSE': edt_nse, 'CC': edt_cc},
    'RF': {'RMSE': rf_rmse, 'NSE': rf_nse, 'CC': rf_cc},
    'BRF': {'RMSE': brf_rmse, 'NSE': brf_nse, 'CC': brf_cc},
    'Bi-LSTM': {'RMSE': bilstm_rmse, 'NSE': bilstm_nse, 'CC': bilstm_cc},
    'WBRF': {'RMSE': wbrf_rmse, 'NSE': wbrf_nse, 'CC': wbrf_cc},
    'WBi-LSTM': {'RMSE': wbilstm_rmse, 'NSE': wbilstm_nse, 'CC': wbilstm_cc}
}
print("\n=== Model Comparison on Test Data ===")
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE = {metrics['RMSE']:.4f}, NSE = {metrics['NSE']:.4f}, CC = {metrics['CC']:.4f}")

######################################
# 16. Plot Test Predictions for All Methods (approx. 1-year test period)
######################################
plt.figure(figsize=(14,12))
plot_order = ['LSTM','SVR','MLP','EDT','RF','BRF','Bi-LSTM','WBRF','WBi-LSTM']
for i, model_name in enumerate(plot_order):
    if model_name in ['WBRF','WBi-LSTM']:
        if model_name=='WBRF':
            pred = wbrf_pred
        else:
            pred = wbilstm_pred
        true_vals = y_test_wave
    else:
        if model_name=='LSTM':
            pred = lstm_pred
        elif model_name=='SVR':
            pred = svr_pred
        elif model_name=='MLP':
            pred = mlp_pred
        elif model_name=='EDT':
            pred = edt_pred
        elif model_name=='RF':
            pred = rf_pred
        elif model_name=='BRF':
            pred = brf_pred
        elif model_name=='Bi-LSTM':
            pred = bilstm_pred
        true_vals = y_test
    plt.subplot(5,2,i+1)
    plt.plot(true_vals, label='actual sppi', marker='o', markersize=3)
    plt.plot(pred, label=model_name, marker='x', linestyle='--')
    plt.title(model_name, fontsize=10)
    # plt.xlabel('Test Index')
    plt.ylabel('SPI')
    plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

######################################
# 17. Taylor Diagram for Model Comparison
######################################
# Use the approx-based predictions for models not wavelet-boosted.
obs_std = np.std(y_test)
model_names_td = ['LSTM','SVR','MLP','EDT','RF','BRF','Bi-LSTM']
std_devs = [np.std(lstm_pred), np.std(svr_pred), np.std(mlp_pred), np.std(edt_pred), np.std(rf_pred), np.std(brf_pred), np.std(bilstm_pred)]
corrs = [lstm_cc, svr_cc, mlp_cc, edt_cc, rf_cc, brf_cc, bilstm_cc]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, polar=True)
theta = [np.arccos(c) for c in corrs]
r = std_devs
for i, name in enumerate(model_names_td):
    ax.plot(theta[i], r[i], 'o', label=name)
# Plot observation as reference
ax.plot(0, obs_std, 'r*', markersize=12, label='Obs')
ax.set_title('Taylor Diagram', fontsize=14)
ax.set_rlim(0, max(std_devs + [obs_std])*1.1)
corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
theta_ticks = [np.arccos(c) for c in corr_ticks]
ax.set_thetagrids(np.degrees(theta_ticks), labels=[str(c) for c in corr_ticks])
# Place legend outside the plot area
plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=8)
plt.show()
######################################
# 18. Forecast 10 Years (120 Months) into the Future using the Best Model
######################################
# Select best model based on RMSE among all nine methods.
model_rmse_dict = {name: metrics['RMSE'] for name, metrics in results.items()}
best_model_name = min(model_rmse_dict, key=model_rmse_dict.get)
print(f"\nBest model selected for forecasting: {best_model_name}")

n_forecast = 120
forecast = []

if best_model_name in ['WBRF', 'WBi-LSTM']:
    # For wavelet-boosted models, maintain two separate lists:
    # current_approx and current_detail, each of length = len(selected_lags)
    current_approx = approx[-len(selected_lags):].tolist()  # last len(selected_lags) values from approx
    current_detail = detail[-len(selected_lags):].tolist()  # last len(selected_lags) values from detail
    
    for i in range(n_forecast):
        # Build input: shape (1, len(selected_lags), 2)
        X_input = np.array(np.column_stack((current_approx, current_detail)))
        X_tensor = torch.tensor(X_input.reshape(1, len(selected_lags), 2), dtype=torch.float32)
        if best_model_name == 'WBi-LSTM':
            with torch.no_grad():
                next_val = wbilstm_model(X_tensor).item()
        elif best_model_name == 'WBRF':
            next_val = wbrf_model.predict(X_input.reshape(1, -1))[0]
        forecast.append(next_val)
        # Update current_approx: drop first and append predicted next approx value.
        current_approx.pop(0)
        current_approx.append(next_val)
        # For current_detail, assume persistence (append last known detail value)
        current_detail.pop(0)
        current_detail.append(current_detail[-1])
else:
    # For non-wavelet-boosted methods, use processed_series and its lags.
    base_series = processed_series
    current_input = base_series[-max(selected_lags):].tolist()  # length = max(selected_lags)
    for i in range(n_forecast):
        X_input = np.array(current_input).astype('float32')
        if best_model_name == 'LSTM':
            X_tensor = torch.tensor(X_input.reshape(1, len(current_input), 1), dtype=torch.float32)
            with torch.no_grad():
                next_val = lstm_model(X_tensor).item()
        elif best_model_name == 'SVR':
            next_val = svr_model.predict(X_input.reshape(1, -1))[0]
        elif best_model_name == 'MLP':
            X_tensor = torch.tensor(X_input.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                next_val = mlp_model(X_tensor).item()
        elif best_model_name == 'EDT':
            next_val = edt_model.predict(X_input.reshape(1, -1))[0]
        elif best_model_name == 'RF':
            next_val = rf_model.predict(X_input.reshape(1, -1))[0]
        elif best_model_name == 'BRF':
            n_boot_fore = 30
            preds = 0
            for j in range(n_boot_fore):
                idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train[idx]
                y_boot = y_train[idx]
                temp_model = RandomForestRegressor(n_estimators=100, random_state=42+j)
                temp_model.fit(X_boot, y_boot)
                preds += temp_model.predict(X_input.reshape(1, -1))[0]
            next_val = preds / n_boot_fore
        elif best_model_name == 'Bi-LSTM':
            X_tensor = torch.tensor(X_input.reshape(1, len(current_input), 1), dtype=torch.float32)
            with torch.no_grad():
                next_val = bi_lstm_model(X_tensor).item()
        forecast.append(next_val)
        current_input.pop(0)
        current_input.append(next_val)

# Plot the forecast along with the historical series.
plt.figure(figsize=(10,5))
if best_model_name in ['WBRF', 'WBi-LSTM']:
    base_series = approx  # target is approx for wavelet-boosted models
else:
    base_series = processed_series
plt.plot(range(len(base_series)), base_series, label='Historical Series')
plt.plot(range(len(base_series), len(base_series)+n_forecast), forecast, label='10-Year Forecast', color='red', marker='o', linestyle='--')
plt.title(f'10-Year Forecast using {best_model_name}')
plt.xlabel('Time Index (months)')
plt.ylabel('SPI')
plt.legend()
plt.show()

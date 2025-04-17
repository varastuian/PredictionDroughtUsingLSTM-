import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet  # keep if you still want to use it elsewhere
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler  # using StandardScaler now
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from scipy.stats import gamma, norm
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ========================
# --- Step 1: Load data ---
# ========================
df = pd.read_csv("./result/merged_data.csv")
df['data'] = pd.to_datetime(df['data'])
df['month'] = df['data'].dt.to_period('M')
df = df.sort_values(by=['station_id', 'data'])
df.head()
# Group by station and month, summing precipitation
monthly_precip = df.groupby(['station_id', 'month'])['rrr24'].sum().reset_index()
monthly_precip['month'] = monthly_precip['month'].dt.to_timestamp()
monthly_precip.rename(columns={'month': 'ds', 'rrr24': 'precip'}, inplace=True)
# monthly_precip.head()

def compute_spi(precip_series):
    precip_array = precip_series.values
    nonzero = precip_array[precip_array > 0]

    if len(nonzero) == 0:
        return np.full_like(precip_array, np.nan, dtype=float)

    # Fit gamma distribution to non-zero data
    shape, loc, scale_param = gamma.fit(nonzero, floc=0)

    # Zero probability
    zero_prob = (precip_array == 0).sum() / len(precip_array)

    # Compute SPI values
    spi_values = np.full_like(precip_array, np.nan, dtype=float)
    for i, val in enumerate(precip_array):
        if np.isnan(val):
            continue
        if val == 0:
            prob = zero_prob
        else:
            prob = zero_prob + (1 - zero_prob) * gamma.cdf(val, shape, loc, scale_param)
        # Clip to avoid extreme ppf values at 0 or 1
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        spi_values[i] = norm.ppf(prob)

    return spi_values

def compute_spi_timescales(data, col='rrr24', timescales=[1, 3, 6, 9, 12, 24]):
    spi_results = {}
    
    # for scale in timescales:
    for scale in [3]:
        # Rolling precipitation over the given scale
        rolling_precip = data[col].rolling(scale, min_periods=scale).sum()

        # Compute SPI for the rolling sum
        spi_array = compute_spi(rolling_precip)

        # Assign back to a DataFrame aligned with original index
        spi_series = pd.Series(spi_array, index=data.index, name=f"SPI_{scale}")
        spi_results[f"SPI_{scale}"] = spi_series

    return pd.DataFrame(spi_results)

all_spi_data = {}

# for station_id in monthly_precip['station_id'].unique():
for station_id in [40708]:
    station_data = monthly_precip[monthly_precip['station_id'] == station_id].copy()
    station_data = station_data.sort_values('ds').reset_index(drop=True)

    spi_df = compute_spi_timescales(station_data, col='precip')
    combined = pd.concat([station_data[['ds']], spi_df], axis=1)
    
    all_spi_data[station_id] = combined

# ============================
# --- Step 4: Prepare data for LSTM ---
# ============================
# Get the continuous SPI series (dropping NaNs)
# spi_series1 = station_data['spi'].dropna().values.reshape(-1, 1)
# spi_series = all_spi_data[40708]['SPI_12'].dropna().values.reshape(-1, 1)
# spi_date = all_spi_data[40708].loc[all_spi_data[40708]['SPI_12'].notna()==True, 'ds'].values


valid_rows = all_spi_data[40708]['SPI_3'].notna()
spi_series = all_spi_data[40708].loc[valid_rows, 'SPI_3'].values.reshape(-1, 1)
spi_date = all_spi_data[40708].loc[valid_rows, 'ds'].values.reshape(-1, 1) 

# # Use StandardScaler to standardize data (zero mean, unit variance)
# scaler = StandardScaler()
# spi_scaled = scaler.fit_transform(spi_series)

scaler = MinMaxScaler(feature_range=(-1, 1))   # or (-1, 1) if you want negative values
spi_scaled = scaler.fit_transform(spi_series)


# --- Create sequences (for one-step forecasting; will be used in direct training below) ---
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

# Choose a window size that is appropriate (e.g., 24 months)
window_size = 12
X_all, y_all = create_sequences(spi_scaled, window_size)

# Use a continuous portion for scheduled sampling multi‐step training:
# (We will extract one long continuous training sequence from the scaled data)
train_size = int(len(spi_scaled) * 0.8)
# Ensure that the training sequence is continuous:
train_seq = spi_scaled[:train_size + window_size]  # first window_size for input and the rest as target

# Prepare tensors for scheduled sampling training:
input_seq = torch.tensor(train_seq[:window_size].reshape(1, window_size, 1), dtype=torch.float32)
# The target is the next "forecast_horizon" values – here we use all available targets from the training period.
target_seq = torch.tensor(train_seq[window_size:], dtype=torch.float32)  # shape: (forecast_horizon, 1)

# ============================
# --- Step 5: Define LSTM Models ---
# ============================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
    
class StackedLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(StackedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# You can choose either model. Here we use the stacked version.
model = StackedLSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =======================================
# --- Step 6: Scheduled Sampling Training ---
# =======================================
import torch
import numpy as np

def train_with_scheduled_sampling(
    model, input_seq, target_seq, optimizer, loss_fn, 
    epochs=150, initial_teacher_forcing_ratio=1.0, 
    min_teacher_forcing_ratio=0.1, decay=0.98, warmup_epochs=30, 
    clip_grad=True, max_grad_norm=1.0,
    early_stop=True, patience=10
):
    model.train()
    forecast_horizon = target_seq.size(0)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Early stopping state
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Teacher forcing ratio schedule
        if epoch < warmup_epochs:
            teacher_forcing_ratio = initial_teacher_forcing_ratio
        else:
            decay_steps = epoch - warmup_epochs
            teacher_forcing_ratio = max(initial_teacher_forcing_ratio * (decay ** decay_steps), min_teacher_forcing_ratio)

        current_input = input_seq.clone()
        outputs = []

        for t in range(forecast_horizon):
            pred = model(current_input)
            outputs.append(pred)

            if np.random.rand() < teacher_forcing_ratio:
                next_val = target_seq[t].view(1, 1, 1)
            else:
                with torch.no_grad():
                    next_val = pred.view(1, 1, 1)
                    # Optionally add noise for robustness
                    # next_val += 0.01 * torch.randn_like(next_val)

            current_input = torch.cat([current_input[:, 1:, :], next_val], dim=1)

        outputs = torch.cat(outputs, dim=0)
        loss = loss_fn(outputs, target_seq)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")

        # Early stopping logic
        if early_stop:
            if loss.item() < best_loss - 1e-4:  # use delta to avoid tiny fluctuation resets
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} | Best Loss: {best_loss:.4f}")
                    break

# Call the scheduled sampling training:
# train_with_scheduled_sampling(model, input_seq, target_seq, optimizer, loss_fn, epochs=150)
train_with_scheduled_sampling(model, input_seq, target_seq, optimizer, loss_fn, early_stop=True, patience=15)

# =======================================
# --- Step 7: Evaluate on Test Data (One-step Prediction) ---
# =======================================
# For evaluating one-step predictions we can use the standard splitting from our sequences:
train_split = int(0.8 * len(X_all))
X_train, X_test = X_all[:train_split], X_all[train_split:]
y_train, y_test = y_all[:train_split], y_all[train_split:]

# Train the model in one-step mode for comparison if desired:
# (Note: this loop is separate from scheduled sampling training.)
model.eval()
for epoch in range(150):
    model.train()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"One-step Training Epoch {epoch}, Loss: {loss.item():.4f}")

# model.eval()
# with torch.no_grad():
#     y_pred = model(X_test)
# y_pred_inv = scaler.inverse_transform(y_pred.numpy())
# y_test_inv = scaler.inverse_transform(y_test.numpy())

# # Plot actual vs one-step predictions:
# date_series = pd.to_datetime(station_data['ds'].dropna().reset_index(drop=True))
# forecast_start_index = len(date_series) - len(y_test_inv)
# forecast_dates = date_series[forecast_start_index:]

# plt.figure(figsize=(10, 5))
# plt.plot(forecast_dates, y_test_inv, label="Actual SPI (Next 12 months)")
# plt.plot(forecast_dates, y_pred_inv, label="Predicted SPI (LSTM one-step)")
# plt.title("SPI Forecast with PyTorch LSTM (One-step)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# =======================================
# --- Step 8: Forecast Next 10 Years with Recursive Forecasting ---
# =======================================
# Here we use the trained model to forecast future values recursively. We start from the last window in the data.
future_steps = 312 # forecast horizon (months)
last_sequence = spi_scaled[-window_size:]
current_seq = torch.tensor(last_sequence.reshape(1, window_size, 1), dtype=torch.float32)

predicted_future = []
model.eval()
with torch.no_grad():
    for _ in range(future_steps):
        next_val = model(current_seq)
        predicted_future.append(next_val.item())
        # Update the sequence with the new prediction
        new_seq = np.append(current_seq.numpy().squeeze(0)[1:], [[next_val.item()]], axis=0)
        current_seq = torch.tensor(new_seq.reshape(1, window_size, 1), dtype=torch.float32)

# Inverse transform predictions (using StandardScaler)
future_predictions_scaled = np.array(predicted_future).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions_scaled)

# Generate future dates from the last date in your observed series
# last_date = pd.to_datetime(spi_date[-1])
last_date = pd.Timestamp(spi_date[-1][0])
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')

plt.figure(figsize=(12, 6))
plt.plot(spi_date, spi_series, label="Observed SPI", color='blue')
plt.plot(future_dates, future_predictions, label="Predicted SPI (Next 10 Years)", color='green')
plt.axhline(-1, color='orange', linestyle='--', label='Moderate Drought')
plt.axhline(-1.5, color='red', linestyle='--', label='Severe Drought')
plt.axhline(-2, color='purple', linestyle='--', label='Extreme Drought')
plt.title(f"SPI Forecast for Station {station_id} - Next 10 Years")
plt.xlabel("Date")
plt.ylabel("SPI Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate one-step predictions error for reference
# mse = mean_squared_error(y_test_inv, y_pred_inv)
# print(f"One-step Forecast MSE: {mse:.4f}")
# import time
# plt.savefig(f"forcast{int(time.time())}")
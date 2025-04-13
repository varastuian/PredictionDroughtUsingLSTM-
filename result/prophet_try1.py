import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet  # keep if you still want to use it elsewhere
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler  # using StandardScaler now
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ========================
# --- Step 1: Load data ---
# ========================
df = pd.read_csv("result/merged_data.csv")  
df['data'] = pd.to_datetime(df['data'])
df['month'] = df['data'].dt.to_period('M')
df = df.sort_values(by=['station_id', 'data'])

# ============================
# --- Step 2: Preprocessing ---
# ============================
monthly_precip = df.groupby(['station_id', 'month'])['rrr24'].sum().reset_index()
monthly_precip['month'] = monthly_precip['month'].dt.to_timestamp()
monthly_precip.rename(columns={'month': 'ds', 'rrr24': 'precip'}, inplace=True)

# ============================
# --- Step 3: Compute SPI ---
# ============================
def compute_spi(series, scale=12):
    """
    Compute Standardized Precipitation Index (SPI) over a rolling window.
    """
    spi = pd.Series(index=series.index, dtype='float64')
    for i in range(scale, len(series)):
        window = series[i-scale:i]
        if window.std() != 0:
            spi.iloc[i] = (window.iloc[-1] - window.mean()) / window.std()
        else:
            spi.iloc[i] = 0
    return spi

# Use a single station for demonstration
station_id = 40708  
station_data = monthly_precip[monthly_precip['station_id'] == station_id].copy()

# Compute SPI-12 (12 month scale)
station_data['spi'] = compute_spi(station_data['precip'], scale=12)

# Drop NaNs and rename for Prophet (if needed later)
spi_df = station_data.dropna(subset=['spi'])[['ds', 'spi']].copy()
spi_df.rename(columns={'spi': 'y'}, inplace=True)

# ============================
# --- Save DataFrames (if needed)
# ============================
df.to_csv("main_data.csv", index=False)
monthly_precip.to_csv("monthly_precipitation.csv", index=False)
spi_df.to_csv("spi_data.csv", index=False)

# ============================
# --- Step 4: Prepare data for LSTM ---
# ============================
# Get the continuous SPI series (dropping NaNs)
spi_series = station_data['spi'].dropna().values.reshape(-1, 1)

# Use StandardScaler to standardize data (zero mean, unit variance)
scaler = StandardScaler()
spi_scaled = scaler.fit_transform(spi_series)

# --- Create sequences (for one-step forecasting; will be used in direct training below) ---
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

# Choose a window size that is appropriate (e.g., 24 months)
window_size = 24
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
def train_with_scheduled_sampling(model, input_seq, target_seq, optimizer, loss_fn, 
                                  epochs=150, initial_teacher_forcing_ratio=1.0, 
                                  min_teacher_forcing_ratio=0.1, decay=0.98):
    """
    Train the model on a continuous sequence using scheduled sampling.
    input_seq: tensor of shape (1, window_size, 1) – the initial window.
    target_seq: tensor of shape (T, 1) – the ground truth values for T future steps.
    """
    model.train()
    forecast_horizon = target_seq.size(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Compute current teacher forcing ratio (decays with epoch)
        teacher_forcing_ratio = max(initial_teacher_forcing_ratio * (decay ** epoch), min_teacher_forcing_ratio)
        # Reset the current input for this epoch:
        current_input = input_seq.clone()  # shape: (1, window_size, 1)
        outputs = []
        # Generate a forecast for each time step in the training target_seq:
        for t in range(forecast_horizon):
            # Get model prediction for the next time step
            pred = model(current_input)  # shape: (1, 1)
            outputs.append(pred)
            # Decide whether to use the ground truth value or the model prediction
            if np.random.rand() < teacher_forcing_ratio:
                # Use ground truth, reshape to (1, 1, 1)
                next_val = target_seq[t].view(1, 1, 1)
            else:
                # Use model prediction (detach to avoid backprop through it twice)
                next_val = pred.view(1, 1, 1)
            # Slide the window: discard the first element, append the chosen next_val
            current_input = torch.cat([current_input[:, 1:, :], next_val], dim=1)
        # Stack the outputs: results in shape (forecast_horizon, 1)
        outputs = torch.cat(outputs, dim=0)
        loss = loss_fn(outputs, target_seq)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")

# Call the scheduled sampling training:
train_with_scheduled_sampling(model, input_seq, target_seq, optimizer, loss_fn, epochs=150)

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

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
y_pred_inv = scaler.inverse_transform(y_pred.numpy())
y_test_inv = scaler.inverse_transform(y_test.numpy())

# Plot actual vs one-step predictions:
date_series = pd.to_datetime(station_data['ds'].dropna().reset_index(drop=True))
forecast_start_index = len(date_series) - len(y_test_inv)
forecast_dates = date_series[forecast_start_index:]

plt.figure(figsize=(10, 5))
plt.plot(forecast_dates, y_test_inv, label="Actual SPI (Next 12 months)")
plt.plot(forecast_dates, y_pred_inv, label="Predicted SPI (LSTM one-step)")
plt.title("SPI Forecast with PyTorch LSTM (One-step)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================================
# --- Step 8: Forecast Next 10 Years with Recursive Forecasting ---
# =======================================
# Here we use the trained model to forecast future values recursively. We start from the last window in the data.
future_steps = 160  # forecast horizon (months)
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
last_date = pd.to_datetime(station_data['ds'].iloc[-1])
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='M')

plt.figure(figsize=(12, 6))
plt.plot(station_data['ds'], station_data['spi'], label="Observed SPI", color='blue')
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
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f"One-step Forecast MSE: {mse:.4f}")

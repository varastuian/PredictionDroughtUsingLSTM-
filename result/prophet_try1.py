import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import torch
import torch.nn as nn

# --- Step 1: Load data ---
df = pd.read_csv("result\merged_data.csv")  # replace with your actual file path

# --- Step 2: Preprocessing ---
df['data'] = pd.to_datetime(df['data'])
df['month'] = df['data'].dt.to_period('M')
df = df.sort_values(by=['station_id', 'data'])


# --- Step 3: Monthly precipitation per station ---
monthly_precip = df.groupby(['station_id', 'month'])['rrr24'].sum().reset_index()
monthly_precip['month'] = monthly_precip['month'].dt.to_timestamp()

# Rename columns for Prophet
monthly_precip.rename(columns={'month': 'ds', 'rrr24': 'precip'}, inplace=True)

# --- Step 4: Compute SPI function ---
def compute_spi(series, scale=12):
    """
    Compute Standardized Precipitation Index (SPI)
    """
    spi = pd.Series(index=series.index, dtype='float64')
    for i in range(scale, len(series)):
        window = series[i-scale:i]
        if window.std() != 0:
            spi.iloc[i] = (window.iloc[-1] - window.mean()) / window.std()
        else:
            spi.iloc[i] = 0
    return spi

# We'll demonstrate with one station
station_id = 40708  
station_data = monthly_precip[monthly_precip['station_id'] == station_id].copy()

# Compute SPI-12 (12 month scale)
station_data['spi'] = compute_spi(station_data['precip'], scale=12)

# Drop NaNs for modeling
spi_df = station_data.dropna(subset=['spi'])[['ds', 'spi']].copy()
spi_df.rename(columns={'spi': 'y'}, inplace=True)


# Save main DataFrame
df.to_csv("main_data.csv", index=False)

# Save monthly precipitation
monthly_precip.to_csv("monthly_precipitation.csv", index=False)

# Save SPI data
spi_df.to_csv("spi_data.csv", index=False)





# # --- Step 5: Forecast using Prophet ---
# model = Prophet()
# model.fit(spi_df)

# future = model.make_future_dataframe(periods=12, freq='M')
# forecast = model.predict(future)

# # --- Step 6: Plot SPI forecast ---
# plt.figure(figsize=(12, 6))
# plt.plot(spi_df['ds'], spi_df['y'], label='SPI (observed)')
# plt.plot(forecast['ds'], forecast['yhat'], label='SPI Forecast')
# plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
# plt.axhline(-1, color='orange', linestyle='--', label='Moderate Drought')
# plt.axhline(-1.5, color='red', linestyle='--', label='Severe Drought')
# plt.axhline(-2, color='purple', linestyle='--', label='Extreme Drought')
# plt.title(f'SPI Forecast for Station {station_id}')
# plt.xlabel('Date')
# plt.ylabel('SPI Value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#------step6:LSTM
spi_series = station_data['spi'].dropna().values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
spi_scaled = scaler.fit_transform(spi_series)

# --- Create sequences ---
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

window_size = 12
X, y = create_sequences(spi_scaled, window_size)

# --- Train/Test split ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- Define LSTM model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Train the model ---
epochs = 50
for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Predict ---
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Inverse transform
y_pred_inv = scaler.inverse_transform(y_pred.numpy())
y_test_inv = scaler.inverse_transform(y_test.numpy())
date_series = pd.to_datetime(station_data['data'].dropna().reset_index(drop=True))
forecast_start_index = len(date_series) - len(y_test_inv) + (len(y_test_inv) - 1)
forecast_dates = pd.date_range(
    start=date_series[forecast_start_index + 12], 
    periods=1, 
    freq='MS'  # Month Start
)

# --- Updated plot with X = dates ---
plt.figure(figsize=(10, 5))
plt.plot(forecast_dates, y_test_inv[-1], label="Actual SPI (next 12 months)")
plt.plot(forecast_dates, y_pred_inv[-1], label="Predicted SPI (LSTM)")
plt.title("SPI Forecast with PyTorch LSTM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Evaluate ---
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f"MSE: {mse:.4f}")
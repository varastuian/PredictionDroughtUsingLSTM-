import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import pywt   

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import rmse

# -----------------------------
# Settings
# -----------------------------
SEED = 42
window_size = 36
num_epochs = 300
horizon = 1
station_file = r"Data\python_spi\40710_SPI.csv"
spi_column = 'SPI_3'

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)

# Drop rows where SPI is NaN
df = df.dropna(subset=[spi_column])

# -----------------------------
# Wavelet Transform Denoising
# -----------------------------
# Decompose SPI series using wavelet transform
wavelet = 'db4'     # Daubechies 4, common for hydrological data
level = 1           # decomposition level
coeffs = pywt.wavedec(df[spi_column].values, wavelet=wavelet, level=level)

# Thresholding to denoise (soft threshold on detail coefficients)
threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df)))
coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c
                   for i, c in enumerate(coeffs)]

# Reconstruct denoised signal
denoised = pywt.waverec(coeffs_denoised, wavelet=wavelet)
df['spi_denoised'] = denoised[:len(df)]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
df['spi_denoised_scaled'] = scaler.fit_transform(df[['spi_denoised']])

# Create TimeSeries
series = TimeSeries.from_dataframe(df, 'ds', 'spi_denoised_scaled')

# -----------------------------
# Train / Test Split
# -----------------------------
train, test = series[:-48], series[-48:]

# -----------------------------
# Model (WT-LSTM)
# -----------------------------
model = RNNModel(
    model='LSTM',
    input_chunk_length=window_size,
    output_chunk_length=horizon,
    training_length=window_size,
    n_epochs=num_epochs,
    dropout=0.2,
    hidden_dim=64,
    batch_size=16,
    random_state=SEED,
    model_name=f"SPI_WT_LSTM_40700_{spi_column}"
)

# -----------------------------
# Fit model
# -----------------------------
model.fit(train, verbose=True)

# -----------------------------
# Predict on test
# -----------------------------
# pred = model.predict(len(test))
pred = model.historical_forecasts(
    series,
    start=train.end_time(),  # start predicting right after training set
    forecast_horizon=1,
    stride=1,
    retrain=False,
    verbose=True
)

# test_aligned = test.slice_intersect(pred)
pred = pred.slice_intersect(test)


# Inverse transform predictions & test
# o = scaler.inverse_transform(np.array(test_aligned.values().flatten()).reshape(-1,1)).flatten()
# p = scaler.inverse_transform(np.array(pred.values().flatten()).reshape(-1,1)).flatten()
o = scaler.inverse_transform(test.slice_intersect(pred).values().reshape(-1,1)).flatten()
p = scaler.inverse_transform(pred.values().reshape(-1,1)).flatten()

# -----------------------------
# Metrics
# -----------------------------
print("Lengths -> test:", len(o), " pred:", len(p))

print("RMSE:", rmse(test.slice_intersect(pred), pred))

print("Pearson correlation:", pearsonr(o, p)[0])

# -----------------------------
# Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=1.5, alpha=0.6)
plt.plot(df['ds'], df['spi_denoised'], label="Denoised SPI", lw=2, color="blue")
plt.plot(pred.time_index, p, label="WT-LSTM Predicted", lw=2, color="red", linestyle="--")

plt.title(f"{spi_column} WT-LSTM Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)  # drought line
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Forecast till 2099
# -----------------------------
last_date = df['ds'].max()
months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)

forecast = model.predict(months_to_2099)
forecast_values = scaler.inverse_transform(forecast.values())

plt.figure(figsize=(16,6))
plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=1.5, alpha=0.6)
plt.plot(df['ds'], df['spi_denoised'], label="Denoised SPI", lw=2, color="blue")
plt.plot(forecast.time_index, forecast_values, label="WT-LSTM Forecast", lw=2, color="green", linestyle="--")

plt.title(f"{spi_column} WT-LSTM Forecast till 2099")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.grid(True)
plt.show()

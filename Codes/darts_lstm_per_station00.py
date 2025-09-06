import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers import Scaler

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import rmse

# -----------------------------
# Settings
# -----------------------------
station_file = r"Data\python_spi\40706.csv"
spi_column = 'SPI_3'

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)


# Drop rows where SPI is NaN
df = df.dropna(subset=[spi_column])



# scaler = StandardScaler()
# df[spi_column + "_scaled"] = scaler.fit_transform(df[[spi_column]])
# series = TimeSeries.from_dataframe(df, 'ds', spi_column + "_scaled")

scaler = Scaler()
series = scaler.fit_transform(TimeSeries.from_dataframe(df, 'ds', spi_column))
train, test = series.split_before(0.9)

model = RNNModel(
    model='LSTM',
    input_chunk_length=36,
    output_chunk_length=12,
    training_length=36,
    n_epochs=300,
    dropout=0.2,
    hidden_dim=64,
    batch_size=16,
    random_state=42
)

# -----------------------------
# Fit model
# -----------------------------
model.fit(train, verbose=True)

# -----------------------------
# Predict on test
# -----------------------------
pred = model.predict(len(test))

# o = scaler.inverse_transform(np.array(test.values().flatten()).reshape(-1,1)).flatten()
# p = scaler.inverse_transform(np.array(pred.values().flatten()).reshape(-1,1)).flatten()
o = np.array(test.values().flatten())
p = np.array(pred.values().flatten())
# -----------------------------
# Compute metrics
# -----------------------------
print("RMSE:", rmse(test, pred))
print("Pearson correlation:", pearsonr(o, p)[0])

# -----------------------------
# Plot actual vs predicted
# -----------------------------
plt.figure(figsize=(14,6))
# series.plot(label="Actual", lw=2)
# pred.plot(label="Predicted", lw=2, color="red", linestyle="--")
plt.plot(df['ds'], df[spi_column], label="Actual", lw=2)

plt.plot(pred.time_index, p, label="Predicted", lw=2, color="red", linestyle="--")

plt.title(f"{spi_column} LSTM Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)  # drought line
plt.legend()
plt.grid(True)
plt.show()

# # -----------------------------
# # Forecast till 2099
# # -----------------------------
# #TODO : fit again on full timeseries instead of just train part
# last_date = df['ds'].max()
# months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)

# forecast = model.predict(months_to_2099)
# forecast_values = scaler.inverse_transform(forecast.values())

# plt.figure(figsize=(16,6))
# # series.plot(label="Historical", lw=2)
# # forecast.plot(label="Forecast", lw=2, color="green", linestyle="--")
# plt.plot(df['ds'], df[spi_column], label="Historical", lw=2)
# plt.plot(forecast.time_index, forecast_values, label="Forecast", lw=2, color="green", linestyle="--")
# plt.title(f"{spi_column} LSTM Forecast till 2099")
# plt.title(f"{spi_column} LSTM Forecast till 2099")
# plt.xlabel("Date")
# plt.ylabel(spi_column)
# plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
# plt.legend()
# plt.grid(True)
# plt.show()

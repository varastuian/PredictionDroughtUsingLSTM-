import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import rmse
from darts.metrics import mae, rmse, mape

# -----------------------------
# Settings
# -----------------------------
SEED = 42
window_size = 36
horizon = 12
station_file = r"Data\python_spi\40700.csv"
spi_column = 'SPI_12'
output_folder = "./Results/r8_300epoch"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)


# Drop rows where SPI is NaN
df = df.dropna(subset=[spi_column])



scaler = StandardScaler()
df[spi_column + "_scaled"] = scaler.fit_transform(df[[spi_column]])


# Create TimeSeries
# series = TimeSeries.from_dataframe(df, 'ds', spi_column)
series = TimeSeries.from_dataframe(df, 'ds', spi_column + "_scaled")

# -----------------------------
# Split train/test
# -----------------------------
train, test = series[:-48], series[-48:]
# train, test = series.split_before(0.8)

# -----------------------------
# Model
# -----------------------------
model = RNNModel(
    model='LSTM',
    input_chunk_length=window_size,
    output_chunk_length=horizon,
    training_length=window_size,
    n_epochs=300,
    dropout=0.2,
    hidden_dim=64,
    batch_size=16,
    random_state=SEED,
    model_name=f"SPI_LSTM_40700_{spi_column}"
)

# -----------------------------
# Fit model
# -----------------------------
model.fit(train, verbose=True)

# -----------------------------
# Predict on test
# -----------------------------
pred = model.predict(len(test))

o = scaler.inverse_transform(np.array(test.values().flatten()).reshape(-1,1)).flatten()
p = scaler.inverse_transform(np.array(pred.values().flatten()).reshape(-1,1)).flatten()

# -----------------------------
# Compute metrics
# -----------------------------
mae_val = mae(test, pred)
rmse_val = rmse(test, pred)
mape_val = mape(test, pred)
corr =  pearsonr(o, p)[0]
print("RMSE:", rmse(test, pred))
print("Pearson correlation:", corr)

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
# Add metrics text box
metrics_text = f"MAE: {mae_val:.3f}\nRMSE: {rmse_val:.3f}\nMAPE: {mape_val:.2f}\nCorr: {corr:.3f}"
plt.gca().text(
    0.02, 0.95, metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
)

outfile = os.path.join(output_folder, f"{spi_column}_lstmPredicted.png")
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Forecast till 2099
# -----------------------------
#TODO : fit again on full timeseries instead of just train part
last_date = df['ds'].max()
months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)

forecast = model.predict(months_to_2099)
forecast_values = scaler.inverse_transform(forecast.values())

plt.figure(figsize=(16,6))
# series.plot(label="Historical", lw=2)
# forecast.plot(label="Forecast", lw=2, color="green", linestyle="--")
plt.plot(df['ds'], df[spi_column], label="Historical", lw=2)
plt.plot(forecast.time_index, forecast_values, label="Forecast", lw=2, color="green", linestyle="--")
plt.title(f"{spi_column} LSTM Forecast till 2099")
plt.title(f"{spi_column} LSTM Forecast till 2099")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.grid(True)
# Add metrics text box
metrics_text = f"MAE: {mae_val:.3f}\nRMSE: {rmse_val:.3f}\nMAPE: {mape_val:.2f}\nCorr: {corr:.3f}"
plt.gca().text(
    0.02, 0.95, metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
)
outfile = os.path.join(output_folder, f"{spi_column}_lstmforecast.png")
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

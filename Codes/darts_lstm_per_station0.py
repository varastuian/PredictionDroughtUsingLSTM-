import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from darts.dataprocessing.transformers import Scaler

from darts import TimeSeries
from darts.models import RNNModel,BlockRNNModel
from darts.metrics import rmse
from darts.metrics import mae, rmse, mape
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.likelihood_models import QuantileRegression
import torch

# -----------------------------
# Settings
# -----------------------------
station_file = r"Data\python_spi\40706.csv"
spi_column = 'SPI_12'
output_folder = "./Results/40706"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)
df = df.dropna(subset=[spi_column])



scaler = Scaler()
series = scaler.fit_transform(TimeSeries.from_dataframe(df, 'ds', spi_column))
train, test = series.split_before(0.8)

covariates = TimeSeries.from_dataframe(df, "ds", ["tm_m", "precip"])

cov_scaler = Scaler()
covariates_scaled = cov_scaler.fit_transform(covariates)
train_cov, test_cov = covariates_scaled.split_before(0.8)



model = BlockRNNModel(
    model='LSTM',
    input_chunk_length=36,
    output_chunk_length=6,
    n_epochs=91,
    batch_size=16,
    hidden_dim=64, 
    dropout=0.2,
    random_state=42
)

model.fit(train, val_series=test, past_covariates=train_cov,val_past_covariates=test_cov, verbose=True)
prediction = model.predict(len(test), past_covariates=covariates_scaled)
prediction = scaler.inverse_transform(prediction)
series_test = scaler.inverse_transform(test)
o = np.array(series_test.values().flatten())
p = np.array(prediction.values().flatten())
corr = pearsonr(o, p)[0]
mae_val = mae(series_test, prediction)
rmse_val = rmse(series_test, prediction)
mape_val = mape(series_test, prediction)

model.fit(series, past_covariates=covariates_scaled, verbose=True)

last_date = df['ds'].max()
months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month +1 )

future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                             end="2099-12-01", freq='MS')


last_covariates = covariates_scaled[-model.input_chunk_length:]

future_cov = pd.DataFrame({
    'ds': future_dates,
    'tm_m': np.full(len(future_dates), df['tm_m'].mean()),  # example: mean temperature
    'precip': np.full(len(future_dates), df['precip'].mean())  # example: mean precipitation
})
future_cov_ts = TimeSeries.from_dataframe(future_cov, 'ds', ['tm_m', 'precip'])
future_cov_scaled = cov_scaler.transform(future_cov_ts)

full_future_cov = last_covariates.concatenate(future_cov_scaled)

future_cov_series = cov_scaler.transform(TimeSeries.from_dataframe(future_cov, 'ds', ['tm_m', 'precip']))
forecast = model.predict(
    n=months_to_2099,
    series=series,
    past_covariates=full_future_cov
)
forecast_values = scaler.inverse_transform(forecast)




plt.figure(figsize=(16,6))
plt.plot(df['ds'], df[spi_column], label="Historical", lw=0.6)
plt.plot(prediction.time_index, p, label="Predicted", lw=0.4, color="red", linestyle="--")
plt.plot(forecast.time_index, forecast_values.values(), label="Forecast", lw=0.6, color="green")
plt.title(f"{spi_column} LSTM Forecast till 2099")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.grid(True)
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

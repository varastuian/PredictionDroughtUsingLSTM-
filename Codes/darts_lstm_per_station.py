import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries
from darts.models import RNNModel,BlockRNNModel
from darts.metrics import rmse
from pytorch_lightning.callbacks import EarlyStopping
from darts.dataprocessing.transformers import Scaler

# -----------------------------
# Settings
# -----------------------------
SEED = 42
window_size = 50
num_epochs = 100
horizon = 1
station_file = r"Data\testdata\40706.csv"
spi_column = 'SPI_12'
output_folder = "./Results/r7"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)


# Drop rows where SPI is NaN
df = df.dropna(subset=[spi_column])


early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    mode="min"
)


series = TimeSeries.from_dataframe(df, 'ds', spi_column)
target_scaler = Scaler()
series_scaled = target_scaler.fit_transform(series)
n = -70
train, test = series_scaled[:n], series_scaled[n:]


# month_series = datetime_attribute_timeseries(series.time_index, attribute="month", one_hot=True)
# covariates = covariates.stack(month_series)

covariates = TimeSeries.from_dataframe(df, "ds", ["tm_m", "precip"])
cov_scaler = Scaler()
covariates_scaled = cov_scaler.fit_transform(covariates)
train_cov = covariates_scaled[:len(train)]
val_cov = covariates_scaled[len(train)-window_size:len(train)+len(test)]

model = BlockRNNModel(
    model='LSTM',
    input_chunk_length=window_size,
    output_chunk_length=horizon,
    n_epochs=num_epochs,
    dropout=0.2,
    hidden_dim=64,
    batch_size=16,
    random_state=SEED,
    pl_trainer_kwargs={ "callbacks": [early_stopper], }

)


# model.fit(train, val_series=test, past_covariates=train_cov,val_past_covariates=val_cov, verbose=True)

# pred = model.predict(len(test), past_covariates=val_cov)
# pred = target_scaler.inverse_transform(pred)
# test_orig = target_scaler.inverse_transform(test)
# o = np.array(test_orig.values().flatten())
# p = np.array(pred.values().flatten())
# print("RMSE:", rmse(test, pred))
# print("Pearson correlation:", pearsonr(o, p)[0])




# plt.figure(figsize=(14,6))
# plt.plot(df['ds'], df[spi_column], label="Actual", lw=2)
# plt.plot(pred.time_index, p, label="Predicted", lw=2, color="red", linestyle="--")
# plt.title(f"{spi_column} LSTM Forecast vs Actual")
# plt.xlabel("Date")
# plt.ylabel(spi_column)
# plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)  # drought line
# plt.legend()
# plt.grid(True)
# outfile = os.path.join(output_folder, f"{spi_column}_lstmPredicted.png")
# plt.savefig(outfile, dpi=300, bbox_inches="tight")
# plt.close()


# -----------------------------
# Forecast till 2099
# -----------------------------
last_date = df['ds'].max()
months_to_2099 = (2050 - last_date.year) * 12 + (12 - last_date.month + 1)
print(len(series_scaled))
model = BlockRNNModel(
    model='LSTM',
    input_chunk_length=window_size+30,
    output_chunk_length=horizon+11,
    n_epochs=num_epochs,
    dropout=0.1,
    hidden_dim=64,
    batch_size=16,
    random_state=SEED
)
model.fit(series_scaled ,verbose=True)

forecast = model.predict(months_to_2099)
forecast = target_scaler.inverse_transform(forecast)
forecast_values = forecast.values()
plt.figure(figsize=(16,6))
plt.plot(df['ds'], df[spi_column], label="Historical", lw=2)
plt.plot(forecast.time_index, forecast_values, label="Forecast", lw=2, color="green", linestyle="--")
plt.title(f"{spi_column} LSTM Forecast till 2099")
plt.title(f"{spi_column} LSTM Forecast till 2099")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.grid(True)
outfile = os.path.join(output_folder, f"{spi_column}_lstmForecast.png")
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

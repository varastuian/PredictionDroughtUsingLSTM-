import pandas as pd
import torch
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mae, rmse, mape
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# ===============================
# Load dataset
# ===============================
spi_column = 'SPI_12'

df = pd.read_csv(r"Data\testdata\40706.csv", parse_dates=["ds"])
df = df.sort_values("ds")
df = df.dropna(subset=[spi_column])

# Target series (SPI index)
series = TimeSeries.from_dataframe(df, time_col="ds", value_cols=spi_column)

# Meteorological covariates
covariates = TimeSeries.from_dataframe(df, time_col="ds", value_cols=["precip", "tm_m"])

# Lagged SPI as covariate
lagged_cov = series.shift(1).with_columns_renamed("SPI_12", "lagged_spi")

# ===============================
# Time-based covariates
# ===============================
month_sin = datetime_attribute_timeseries(series.time_index, attribute="month", cyclic=True)
dayofyear_sin = datetime_attribute_timeseries(series.time_index, attribute="day_of_year", cyclic=True)

# ===============================
# Align all series to common index
# ===============================
# common_index = series.time_index.intersection(
#     covariates.time_index
# ).intersection(
#     lagged_cov.time_index
# ).intersection(
#     month_sin.time_index
# ).intersection(
#     dayofyear_sin.time_index
# )

# series = series.slice(common_index[0], common_index[-1])
# covariates = covariates.slice(common_index[0], common_index[-1])
# lagged_cov = lagged_cov.slice(common_index[0], common_index[-1])
# month_sin = month_sin.slice(common_index[0], common_index[-1])
# dayofyear_sin = dayofyear_sin.slice(common_index[0], common_index[-1])

# ===============================
# Stack all covariates
# ===============================
# covariates = covariates.stack(lagged_cov).stack(month_sin).stack(dayofyear_sin)

# ===============================
# Scaling
# ===============================
scaler_target = Scaler()
scaler_cov = Scaler()

series_scaled = scaler_target.fit_transform(series)
cov_scaled = scaler_cov.fit_transform(covariates)

# ===============================
# Train/test split
# ===============================
train, test = series_scaled.split_before(0.8)
train_cov, val_cov = cov_scaled.split_before(0.8)

# ===============================
# Ensure past_covariates extend enough for input_chunk_length
# ===============================
# input_chunk_length = 12

# Extend val_cov backward by input_chunk_length steps
val_cov_extended = cov_scaled.slice(train_cov.start_time(), cov_scaled.end_time())

# ===============================
# Define LSTM model
# ===============================
model = BlockRNNModel(
    model="LSTM",
    input_chunk_length=50,
    output_chunk_length=31,
    # hidden_dim=64,
    # n_rnn_layers=2,
    # dropout=0.2,
    batch_size=32,
    # n_epochs=100,
    n_epochs=91,
    # optimizer_cls=torch.optim.AdamW,
    # optimizer_kwargs={"lr": 1e-4},
    # likelihood=QuantileRegression([0.1, 0.5, 0.9]),
    random_state=42
)

# ===============================
# Fit model
# ===============================
model.fit(
    train,val_series=test,verbose=True
)
# model.fit(
#     train,
#     # ,past_covariates=train_cov,
#     val_series=test
#     # ,val_past_covariates=val_cov,
#     ,verbose=True
# )

# ===============================
# Forecast
# ===============================
forecast = model.predict(
    n=len(test)
    # ,past_covariates=val_cov_extended
)

# Inverse transform
forecast = scaler_target.inverse_transform(forecast)
series_test = scaler_target.inverse_transform(test)
o = np.array(series_test.values().flatten())
p = np.array(forecast.values().flatten())
corr = pearsonr(o, p)[0]

mae_val = mae(series_test, forecast)
rmse_val = rmse(series_test, forecast)
mape_val = mape(series_test, forecast)

print("MAE :", mae_val)
# print("RMSE:", rmse_val)
# print("MAPE:", mape_val)
print("corr:", corr)

# ===============================
# Plot and Save
# ===============================
plt.figure(figsize=(10, 6))
series.plot(label="Actual",linewidth=0.9)
forecast.plot(label="Forecast",linewidth=0.3)
plt.legend()
plt.title("SPI Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("SPI")

# Add metrics text box
metrics_text = f"MAE: {mae_val:.3f}\nRMSE: {rmse_val:.3f}\nMAPE: {mape_val:.2f}\nCorr: {corr:.3f}"
plt.gca().text(
    0.02, 0.95, metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
)

# Save plot
plt.savefig("forecast_result.png", dpi=300, bbox_inches="tight")
plt.close()
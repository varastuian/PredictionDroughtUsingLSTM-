import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import (
    ARIMA, ExponentialSmoothing,
    RegressionModel, BlockRNNModel
)
from darts.dataprocessing.transformers import Scaler

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler
import pywt

# -----------------------------
# Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
window_size = 36
horizon = 1
num_epochs = 100
input_folder = "./Data/testdata"
output_folder = "./Results/r12"
os.makedirs(output_folder, exist_ok=True)

# SPI = ["SPI_12"]
SPI = ["SPI_1","SPI_3","SPI_6","SPI_9","SPI_12","SPI_24"]

# -----------------------------
# Forecast function
# -----------------------------
def train_and_forecast(df, spi, model_name,covariates):
    df_spi = df[["ds", spi]].dropna().reset_index(drop=True)

    # scale if needed
    use_scaler = model_name in ["SVR", "LSTM", "WTLSTM"]
    if use_scaler:
        scaler = StandardScaler()
        df_spi[spi + "_scaled"] = scaler.fit_transform(df_spi[[spi]])
        value_col = spi + "_scaled"
    else:
        scaler = None
        value_col = spi

    series = TimeSeries.from_dataframe(df_spi, "ds", value_col)
    train, test = series[:-48], series[-48:]

    # -----------------------------
    # Model selection
    # -----------------------------
    if model_name == "ARIMA":
        model = ARIMA()
    elif model_name == "ETS":
        model = ExponentialSmoothing()
    elif model_name == "ExtraTrees":
        model = RegressionModel(
            ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
            lags=window_size, output_chunk_length=horizon,lags_future_covariates=[0],
        )
    elif model_name == "RandomForest":
        model = RegressionModel(
            RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
            lags=window_size, output_chunk_length=horizon,lags_future_covariates=[0],
        )
    elif model_name == "SVR":
        model = RegressionModel(
            SVR(kernel="rbf", C=10, gamma="scale"),
            lags=window_size, output_chunk_length=horizon,lags_future_covariates=[0],
        )
    elif model_name == "LSTM":
        model = BlockRNNModel(
            model="LSTM",
            input_chunk_length=window_size,
            output_chunk_length=horizon,
            n_epochs=num_epochs,
            batch_size=16,
            hidden_dim=64,
            n_rnn_layers=2,
            dropout=0.2,
            random_state=SEED
        )
    elif model_name == "WTLSTM":
        coeffs = pywt.wavedec(df_spi[value_col].values, "db4", level=1)
        threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df_spi)))
        coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
        denoised = pywt.waverec(coeffs_denoised, wavelet='db4')
        df_spi["spi_denoised"] = denoised[:len(df_spi)]
        series = TimeSeries.from_dataframe(df_spi, "ds", "spi_denoised")
        train, test = series[:-48], series[-48:]
        model = BlockRNNModel(
            model="LSTM",
            input_chunk_length=window_size,
            output_chunk_length=horizon,
            n_epochs=num_epochs,
            batch_size=16,
            hidden_dim=64,
            n_rnn_layers=2,
            dropout=0.2,
            random_state=SEED
        )

    # -----------------------------
    # Fit + Forecast
    # -----------------------------
    if isinstance(model, RegressionModel):
        model.fit(train, future_covariates=covariates)
        forecast = model.predict(len(test), future_covariates=covariates)

    elif isinstance(model, BlockRNNModel):
        model.fit(train, past_covariates=covariates)
        forecast = model.predict(len(test), past_covariates=covariates)

    else:  # ARIMA, ETS
        model.fit(train)
        forecast = model.predict(len(test))
    # inverse transform if scaled
    if scaler:
        test_vals = scaler.inverse_transform(test.values())
        forecast_vals = scaler.inverse_transform(forecast.values())
    else:
        test_vals = test.values()
        forecast_vals = forecast.values()

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df_spi["ds"], df_spi[spi], label="True", lw=0.8)
    plt.plot(test.time_index, test_vals, label="Test", lw=1, color="blue")
    plt.plot(forecast.time_index, forecast_vals, label=f"{model_name} Forecast", lw=1.2, color="red")
    plt.legend()
    plt.title(f"{spi} Forecast — {model_name}")
    plt.grid(alpha=0.3)

    outfile = os.path.join(output_folder, f"{station}_{spi}_{model_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved {outfile}")

# -----------------------------
# Main Loop
# -----------------------------
for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["ds"])
    covariates = TimeSeries.from_dataframe(df, "ds", ["tm_m", "precip"])

    for spi in SPI:
        for model_name in ["ExtraTrees", "RandomForest", "SVR", "LSTM" ,"WTLSTM","ARIMA"]:
            print(f"--- {station} {spi} {model_name} ---")
            train_and_forecast(df.copy(), spi, model_name,covariates)

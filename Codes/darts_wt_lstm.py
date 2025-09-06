import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import pywt   
from darts.dataprocessing.transformers import Scaler

from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.metrics import rmse,mae, rmse, mape
import glob

# -----------------------------
# Settings
# -----------------------------
input_folder = "./Data/testdata"
# SPI = ["SPI_1","SPI_3","SPI_6","SPI_9","SPI_12","SPI_24"]
SPI = ["SPI_9"]

all_results = []

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station = os.path.splitext(os.path.basename(file))[0]
    if station !="40706":
        continue
    output_folder = f"./Results/wblstm3/{station}"
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(file, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    for spi_column in SPI:
        df = df.dropna(subset=[spi_column])
 
        for col in [spi_column,"tm_m", "precip"]:
            coeffs = pywt.wavedec(df[col].values, wavelet='db4', level=1)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(df)))
            coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c
                            for i, c in enumerate(coeffs)]
            denoised = pywt.waverec(coeffs_denoised, wavelet="db4")
            df[f"{col}_denoised"] = denoised[:len(df)]

        scaler = Scaler()
        series = scaler.fit_transform(TimeSeries.from_dataframe(df, 'ds', f'{spi_column}_denoised'))
        train, test = series.split_before(0.8)

        cov_scaler = Scaler()
        covariates_scaled = cov_scaler.fit_transform(TimeSeries.from_dataframe(df, "ds", ["tm_m_denoised", "precip_denoised"]))
        train_cov, test_cov = covariates_scaled.split_before(0.8)


        # -----------------------------
        # Model (WT-LSTM)
        # -----------------------------
        model = BlockRNNModel(
            model='LSTM',
            input_chunk_length=36,
            output_chunk_length=6,
            n_epochs=91,
            dropout=0.2,
            hidden_dim=64,
            batch_size=16,
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
        std_ref = np.std(o, ddof=1)
        std_sim = np.std(p, ddof=1)
        crmse_val = np.sqrt(std_ref**2 + std_sim**2 - 2*std_ref*std_sim*corr)

        all_results.append({"station": station,
                            "timescale":spi_column,
                            "model": model.model_name,
        "rmse": rmse_val, "corr": corr, "std_ref": std_ref,
        "std_model": std_sim, "crmse": crmse_val})

        
        model.fit(series, past_covariates=covariates_scaled, verbose=True)

        last_date = df['ds'].max()
        months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month +1 )

        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                    end="2099-12-01", freq='MS')


        last_covariates = covariates_scaled[-model.input_chunk_length:]

        future_cov = pd.DataFrame({
            'ds': future_dates,
            'tm_m_denoised': np.full(len(future_dates), df['tm_m_denoised'].mean()),  # example: mean temperature
            'precip_denoised': np.full(len(future_dates), df['precip_denoised'].mean())  # example: mean precipitation
        })
        future_cov_ts = TimeSeries.from_dataframe(future_cov, 'ds', ['tm_m_denoised', 'precip_denoised'])
        future_cov_scaled = cov_scaler.transform(future_cov_ts)

        full_future_cov = last_covariates.concatenate(future_cov_scaled)

        # future_cov_series = cov_scaler.transform(TimeSeries.from_dataframe(future_cov, 'ds', ['tm_m', 'precip']))
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
        outfile = os.path.join(output_folder, f"{spi_column}_lstm.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()
        # Save metrics
        metrics_df = pd.DataFrame(all_results)
        metrics_df.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)

        print("✅ Done! Results saved in:", output_folder)

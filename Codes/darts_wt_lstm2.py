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
from darts.metrics import rmse, mae, mape
import glob

# -----------------------------
# Settings
# -----------------------------
input_folder = "./Data/testdata"
SPI = ["SPI_6","SPI_9","SPI_12"]

all_results = []

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station = os.path.splitext(os.path.basename(file))[0]
    if station != "40706":
        continue
        
    output_folder = f"./Results/wblstm2/{station}"
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(file, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    
    for spi_column in SPI:
        df = df.dropna(subset=[spi_column])
 
        # Denoising (keep your existing denoising code)
        for col in [spi_column, "tm_m", "precip"]:
            coeffs = pywt.wavedec(df[col].values, wavelet='db4', level=1)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(df)))
            coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c
                            for i, c in enumerate(coeffs)]
            denoised = pywt.waverec(coeffs_denoised, wavelet="db4")
            df[f"{col}_denoised"] = denoised[:len(df)]

        # -----------------------------
        # Step 1: Forecast covariates (temperature and precipitation)
        # -----------------------------
        # Prepare covariates data
        cov_cols = ["tm_m_denoised", "precip_denoised"]
        cov_scaler = Scaler()
        covariates_series = cov_scaler.fit_transform(
            TimeSeries.from_dataframe(df, 'ds', cov_cols)
        )
        
        # Train model for covariates forecasting
        cov_model = BlockRNNModel(
            model='LSTM',
            input_chunk_length=36,
            output_chunk_length=12,
            n_epochs=100,
            dropout=0.2,
            hidden_dim=64,
            batch_size=16,
            random_state=42,
            force_reset=True
        )
        
        # Split data for covariates model
        cov_train, cov_val = covariates_series.split_before(0.8)
        cov_model.fit(cov_train, val_series=cov_val, verbose=True)
        
        # Forecast covariates until 2099
        last_date = df['ds'].max()
        months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month)
        
        # Forecast covariates
        cov_forecast = cov_model.predict(
            n=months_to_2099,
            series=covariates_series
        )
        
        # Inverse transform to get actual values
        cov_forecast_actual = cov_scaler.inverse_transform(cov_forecast)
        
        # Combine historical and forecasted covariates
        full_covariates = covariates_series.concatenate(cov_forecast, axis=0)
        
        # -----------------------------
        # Step 2: Forecast SPI using forecasted covariates
        # -----------------------------
        # Prepare SPI data
        spi_scaler = Scaler()
        spi_series = spi_scaler.fit_transform(
            TimeSeries.from_dataframe(df, 'ds', f'{spi_column}_denoised')
        )
        
        # Split data for SPI model
        train, test = spi_series.split_before(0.8)
        train_cov, test_cov = full_covariates.split_before(0.8)
        
        # Train SPI model
        spi_model = BlockRNNModel(
            model='LSTM',
            input_chunk_length=36,
            output_chunk_length=12,
            n_epochs=100,
            dropout=0.2,
            hidden_dim=64,
            batch_size=16,
            random_state=42,
            force_reset=True
        )
        
        spi_model.fit(
            train, 
            past_covariates=train_cov,
            val_series=test,
            val_past_covariates=test_cov,
            verbose=True
        )
        
        # Predict on test period
        prediction = spi_model.predict(
            len(test), 
            past_covariates=full_covariates
        )
        prediction = spi_scaler.inverse_transform(prediction)
        series_test = spi_scaler.inverse_transform(test)
        
        # Calculate metrics
        o = np.array(series_test.values().flatten())
        p = np.array(prediction.values().flatten())
        corr = pearsonr(o, p)[0]
        mae_val = mae(series_test, prediction)
        rmse_val = rmse(series_test, prediction)
        mape_val = mape(series_test, prediction)
        std_ref = np.std(o, ddof=1)
        std_sim = np.std(p, ddof=1)
        crmse_val = np.sqrt(std_ref**2 + std_sim**2 - 2*std_ref*std_sim*corr)

        all_results.append({
            "station": station,
            "timescale": spi_column,
            "model": "LSTM",
            "rmse": rmse_val, 
            "corr": corr, 
            "std_ref": std_ref,
            "std_model": std_sim, 
            "crmse": crmse_val
        })
        
        # Forecast SPI until 2099 using forecasted covariates
        spi_model_final = BlockRNNModel(
            model='LSTM',
            input_chunk_length=36,
            output_chunk_length=12,
            n_epochs=100,
            dropout=0.2,
            hidden_dim=64,
            batch_size=16,
            random_state=42,
            force_reset=True
        )
        
        spi_model_final.fit(
            spi_series, 
            past_covariates=full_covariates[:len(spi_series)],
            verbose=True
        )
        
        # Forecast SPI
        spi_forecast = spi_model_final.predict(
            n=months_to_2099,
            series=spi_series,
            past_covariates=full_covariates
        )
        spi_forecast_actual = spi_scaler.inverse_transform(spi_forecast)
        
        # -----------------------------
        # Visualization
        # -----------------------------
        plt.figure(figsize=(16, 10))
        
        # Plot 1: SPI forecast
        plt.subplot(2, 1, 1)
        plt.plot(df['ds'], df[spi_column], label="Historical", lw=1.0)
        plt.plot(prediction.time_index, p, label="Predicted", lw=1.0, color="red", linestyle="--")
        plt.plot(spi_forecast.time_index, spi_forecast_actual.values(), label="Forecast", lw=1.0, color="green")
        plt.title(f"{spi_column} LSTM Forecast till 2099")
        plt.xlabel("Date")
        plt.ylabel(spi_column)
        plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
        plt.legend()
        plt.grid(True)
        
        metrics_text = f"MAE: {mae_val:.3f}\nRMSE: {rmse_val:.3f}\nMAPE: {mape_val:.2f}%\nCorr: {corr:.3f}"
        plt.gca().text(
            0.02, 0.95, metrics_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
        )
        
        # Plot 2: Covariates forecast
        plt.subplot(2, 1, 2)
        # Historical covariates
        hist_tm = cov_scaler.inverse_transform(covariates_series)["tm_m_denoised"]
        hist_precip = cov_scaler.inverse_transform(covariates_series)["precip_denoised"]
        
        # Forecasted covariates
        forecast_tm = cov_forecast_actual["tm_m_denoised"]
        forecast_precip = cov_forecast_actual["precip_denoised"]
        
        plt.plot(hist_tm.time_index, hist_tm.values(), label="Historical Temp", lw=1.0, color="blue")
        plt.plot(forecast_tm.time_index, forecast_tm.values(), label="Forecasted Temp", lw=1.0, color="lightblue", linestyle="--")
        
        # Create a second y-axis for precipitation
        ax2 = plt.gca().twinx()
        ax2.plot(hist_precip.time_index, hist_precip.values(), label="Historical Precip", lw=1.0, color="red")
        ax2.plot(forecast_precip.time_index, forecast_precip.values(), label="Forecasted Precip", lw=1.0, color="pink", linestyle="--")
        
        plt.title("Temperature and Precipitation Forecast")
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        ax2.set_ylabel("Precipitation")
        
        # Combine legends from both axes
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.gca().legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True)
        plt.tight_layout()
        
        outfile = os.path.join(output_folder, f"{spi_column}_lstm_forecast.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save forecasts to CSV
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months_to_2099,
            freq='MS'
        )
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            f'{spi_column}_forecast': spi_forecast_actual.values().flatten(),
            'tm_forecast': forecast_tm.values().flatten(),
            'precip_forecast': forecast_precip.values().flatten()
        })
        
        forecast_df.to_csv(
            os.path.join(output_folder, f"{spi_column}_forecast_2099.csv"),
            index=False
        )
        
        # Save metrics
        metrics_df = pd.DataFrame(all_results)
        metrics_df.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)

        print("✅ Done! Results saved in:", output_folder)
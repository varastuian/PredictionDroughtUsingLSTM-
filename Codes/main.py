import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import rmse
from darts.models import TFTModel, NBEATSModel, NHiTSModel, TCNModel,RegressionModel, RNNModel



from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import pywt

# -----------------------------
# Global Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
window_size = 12
horizon = 1
num_epochs = 350
# num_epochs = 5
input_folder = "./Data/testdata"
output_folder = "./Results/"
os.makedirs(output_folder, exist_ok=True)

def forecast(df, spi_column, station_name, model_name):
    # results = {}
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, spi_column in enumerate(spi_columns):
        df_spi = df[["ds", spi_column]].dropna().reset_index(drop=True)

        # -----------------------------
        # Decide if we need scaling
        # -----------------------------
        use_scaler = model_name in ["SVR", "LSTM", "WTLSTM", "TFT", "NBEATS", "NHiTS", "TCN"]

        if use_scaler:
            scaler = StandardScaler()

            df_spi[spi_column + "_scaled"] = scaler.fit_transform(df_spi[[spi_column]])
            print('after : ',df_spi.head())
            print(df_spi[spi_column].mean())    
            print(df_spi[spi_column].std())     
            print(df_spi[spi_column + "_scaled"].mean())      # should be ~0
            print(df_spi[spi_column + "_scaled"].std())       # should be ~1


            value_col = spi_column + "_scaled"

        else:
            scaler = None
            value_col = spi_column

        series = TimeSeries.from_dataframe(df_spi, "ds", value_col)
        train, test = series[:-48], series[-48:]

        # -----------------------------
        # Model Selection
        # -----------------------------
        if model_name == "ExtraTrees":
            model = RegressionModel(
                model=ExtraTreesRegressor(
                    n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
                ),
                lags=window_size,
                output_chunk_length=horizon
            )
            
        elif model_name == "TFT":
            model = TFTModel(
                input_chunk_length=window_size,
                output_chunk_length=horizon,
                hidden_size=64,
                lstm_layers=1,
                dropout=0.2,
                batch_size=32,
                n_epochs=num_epochs,
                add_relative_index=True,   # 👈 generates relative time indexes
                add_encoders={
                    "cyclic": {"future": ["month"]},  # 👈 add month as cyclic feature
                    "datetime_attribute": {"future": ["year"]},  # add year
                },
                random_state=SEED
            )

        elif model_name == "NBEATS":
            model = NBEATSModel(
                input_chunk_length=window_size,
                output_chunk_length=horizon,
                n_epochs=num_epochs,
                batch_size=32,
                random_state=SEED
            )

        elif model_name == "NHiTS":
            model = NHiTSModel(
                input_chunk_length=window_size,
                output_chunk_length=horizon,
                n_epochs=num_epochs,
                batch_size=32,
                random_state=SEED
            )

        elif model_name == "TCN":
            model = TCNModel(
                input_chunk_length=window_size,
                output_chunk_length=horizon,
                n_epochs=num_epochs,
                dropout=0.1,
                dilation_base=2,
                num_filters=32,
                kernel_size=3,
                random_state=SEED
            )

        elif model_name == "RandomForest":
            model = RegressionModel(
                model=RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
                ),
                lags=window_size,
                output_chunk_length=horizon
            )

        elif model_name == "SVR":
            model = RegressionModel(
                model=SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
                lags=window_size,
                output_chunk_length=horizon
            )

        elif model_name == "LSTM":
            model = RNNModel(
                model="LSTM",
                input_chunk_length=window_size,
                output_chunk_length=horizon,
                training_length=window_size,
                n_epochs=num_epochs,
                dropout=0.2,
                hidden_dim=64,
                batch_size=16,
                random_state=SEED
            )

        elif model_name == "WTLSTM":
            # Special: wavelet decomposition + multiple LSTMs
            coeffs = pywt.wavedec(df[spi_column].values, "db4", level=1)

            # reconstruct each component back as aligned TimeSeries
            # components = []
            # for i, c in enumerate(coeffs):
            #     reconstructed = pywt.upcoef(
            #         "a" if i == 0 else "d", c, "db4",
            #         level=len(coeffs)-i,
            #         take=len(series)
            #     )
            #     components.append(TimeSeries.from_times_and_values(series.time_index, reconstructed))

            
            # Thresholding to denoise (soft threshold on detail coefficients)
            threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df)))
            coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c
                            for i, c in enumerate(coeffs)]

            # Reconstruct denoised signal
            denoised = pywt.waverec(coeffs_denoised, wavelet='db4')
            df['spi_denoised'] = denoised[:len(df)]

            # -----------------------------
            # Scaling
            # -----------------------------
            scaler = StandardScaler()
            df['spi_denoised_scaled'] = scaler.fit_transform(df[['spi_denoised']])

            # Create TimeSeries
            series = TimeSeries.from_dataframe(df, 'ds', 'spi_denoised_scaled')
            train, test = series[:-48], series[-48:]


            model = RNNModel(
                model='LSTM',
                input_chunk_length=window_size,
                output_chunk_length=horizon,
                training_length=window_size,
                n_epochs=num_epochs,
                dropout=0.2,
                hidden_dim=64,
                batch_size=16,
                random_state=SEED
            )


        # -----------------------------
        # Train + Predict (non-WTLSTM branch)
        # -----------------------------
        model.fit(train)
        # pred = model.predict(len(test), series=train)
        pred = model.historical_forecasts(
        series,
        start=train.end_time(),  # start predicting right after training set
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
        )
        pred = pred.slice_intersect(test)

        if use_scaler :
            o = scaler.inverse_transform(test.slice_intersect(pred).values().reshape(-1, 1)).flatten()
            p = scaler.inverse_transform(pred.values().reshape(-1, 1)).flatten()
        else:
            o = test.values().flatten()
            p = pred.values().flatten()

        rmse_val = rmse(test, pred)
        corr_val = pearsonr(o, p)[0]

        # Forecast till 2099
        last_date = df["ds"].max()
        months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)
        forecast = model.predict(months_to_2099, series=series)

        if use_scaler:
            forecast_values = scaler.inverse_transform(forecast.values())
        else:
            forecast_values = forecast.values()

        ax = axes[i]
        ax.plot(df_spi["ds"], df_spi[spi_column], label="True", lw=0.7, alpha=0.6)
        ax.plot(pred.time_index, p, label="Prediction", lw=1, color="red")
        ax.plot(forecast.time_index, forecast_values, label="Forecast", lw=0.7, color="green", alpha=0.7)
        ax.set_title(f"{spi_column}\nRMSE={rmse_val:.2f}, r={corr_val:.2f}", fontsize=10)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)  # Add legend to each subplot

        if i % 4 == 0:
            ax.set_ylabel("SPI Value")
        if i >= 12:
            ax.set_xlabel("Date")

        results.append({"rmse": rmse_val, "corr": corr_val, "spi": spi_column})

    # --- Save subplot ---
    fig.suptitle(f"{station_name} - {model_name}", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.legend(loc="upper right", fontsize=8)
    plt.savefig(
        os.path.join(output_folder, f"{station_name}_{model_name}_subplot.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    return results




# -----------------------------
# Main Loop: all stations + SPI columns
# -----------------------------
all_results = []
for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["ds"])
    spi_columns = ["SPI_1","SPI_3","SPI_6","SPI_12"]
    print("*"*50)
    print("*"*50)
    print("*"*50)
    print("*"*50)
    print("*"*50)
    for model_name in ["TFT", "NBEATS", "NHiTS", "TCN", "LSTM", "WTLSTM", "ExtraTrees", "RandomForest", "SVR"]:
    # for model_name in ["SVR"]:
        print(f"Running {model_name} on {station_name}")
        print("_"*50)
        print("_"*50)
        print("_"*50)
        print("_"*50)
        print("_"*50)
        results_list = forecast(df.copy(), spi_columns, station_name, model_name)
        for result in results_list:
            result.update({"station": station_name, "model": model_name})
            all_results.append(result)



# Save results summary
pd.DataFrame(all_results).to_csv(
    os.path.join(output_folder, "summary_metrics.csv"), index=False
)
print("✅ Done! Results saved in:", output_folder)

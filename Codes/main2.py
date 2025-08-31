import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import rmse
from darts.models import TFTModel, NBEATSModel, NHiTSModel, TCNModel, RegressionModel, RNNModel

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
window_size = 12
horizon = 1
num_epochs = 350
input_folder = "./Data/testdata"
output_folder = "./Results/r1"
os.makedirs(output_folder, exist_ok=True)


def forecast(df, spi_columns, station_name, model_name):
    results = []

    for spi_column in spi_columns:
        df_spi = df[["ds", spi_column]].dropna().reset_index(drop=True)
        print(f"Running {model_name} on {station_name} - {spi_column}")

        # -----------------------------
        # Scaling
        # -----------------------------
        use_scaler = model_name in ["SVR", "LSTM", "WTLSTM", "TFT", "NBEATS", "NHiTS", "TCN"]
        if use_scaler:
            scaler = StandardScaler()
            df_spi[spi_column + "_scaled"] = scaler.fit_transform(df_spi[[spi_column]])
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
            model = RegressionModel(ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
                                    lags=window_size, output_chunk_length=horizon)

        elif model_name == "RandomForest":
            model = RegressionModel(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
                                    lags=window_size, output_chunk_length=horizon)

        elif model_name == "SVR":
            model = RegressionModel(SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
                                    lags=window_size, output_chunk_length=horizon)

        elif model_name == "LSTM":
            model = RNNModel(model="LSTM", input_chunk_length=window_size, output_chunk_length=horizon,
                             training_length=window_size, n_epochs=num_epochs, dropout=0.2,
                             hidden_dim=64, batch_size=16, random_state=SEED)

        elif model_name == "WTLSTM":
            coeffs = pywt.wavedec(df_spi[value_col].values, "db4", level=1)
            threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(df_spi)))
            coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
            denoised = pywt.waverec(coeffs_denoised, wavelet='db4')
            df_spi['spi_denoised'] = denoised[:len(df_spi)]
            series = TimeSeries.from_dataframe(df_spi, 'ds', 'spi_denoised')
            train, test = series[:-48], series[-48:]
            model = RNNModel(model='LSTM', input_chunk_length=window_size, output_chunk_length=horizon,
                             training_length=window_size, n_epochs=num_epochs, dropout=0.2,
                             hidden_dim=64, batch_size=16, random_state=SEED)

        elif model_name == "TFT":
            model = TFTModel(input_chunk_length=window_size, output_chunk_length=horizon,
                             hidden_size=64, lstm_layers=1, dropout=0.2, batch_size=32, n_epochs=num_epochs,
                             add_relative_index=True,
                             add_encoders={"cyclic": {"future": ["month"]}, "datetime_attribute": {"future": ["year"]}},
                             random_state=SEED)

        elif model_name == "NBEATS":
            model = NBEATSModel(input_chunk_length=window_size, output_chunk_length=horizon,
                                n_epochs=num_epochs, batch_size=32, random_state=SEED)

        elif model_name == "NHiTS":
            model = NHiTSModel(input_chunk_length=window_size, output_chunk_length=horizon,
                               n_epochs=num_epochs, batch_size=32, random_state=SEED)

        elif model_name == "TCN":
            model = TCNModel(input_chunk_length=window_size, output_chunk_length=horizon,
                             n_epochs=num_epochs, dropout=0.1, dilation_base=2,
                             num_filters=32, kernel_size=3, random_state=SEED)

        # -----------------------------
        # Train + Historical Forecasts
        # -----------------------------
        model.fit(train)
        pred = model.historical_forecasts(series, start=train.end_time(), forecast_horizon=1,
                                          stride=1, retrain=False, verbose=False)
        pred = pred.slice_intersect(test)

        if use_scaler:
            o = scaler.inverse_transform(test.slice_intersect(pred).values().reshape(-1, 1)).flatten()
            p = scaler.inverse_transform(pred.values().reshape(-1, 1)).flatten()
        else:
            o = test.values().flatten()
            p = pred.values().flatten()

        rmse_val = rmse(test, pred)
        corr_val = pearsonr(o, p)[0]
        std_ref = np.std(o, ddof=1)
        std_sim = np.std(p, ddof=1)
        crmse_val = np.sqrt(std_ref**2 + std_sim**2 - 2*std_ref*std_sim*corr_val)

        results.append({"rmse": rmse_val, "corr": corr_val, "std_ref": std_ref,
                        "std_model": std_sim, "crmse": crmse_val, "spi": spi_column})
    return results


def taylor_diagram_panel(metrics_df, station, outfile):
    """ Create 2x2 Taylor diagrams for SPI_1, SPI_3, SPI_6, SPI_12 """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'polar': True})
    axes = axes.flatten()

    spi_list = ["SPI_1", "SPI_3", "SPI_6", "SPI_12"]
    for ax, spi in zip(axes, spi_list):
        subset = metrics_df[(metrics_df["station"] == station) & (metrics_df["spi"] == spi)]
        if subset.empty:
            continue

        std_ref = subset["std_ref"].iloc[0]

        # Configure polar axis
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)
        ax.set_thetagrids(np.arange(0, 181, 30), labels=["1.0", "0.87", "0.5", "0", "-0.5", "-1"])
        ax.set_rlabel_position(135)

        # Plot reference
        ax.plot(0, std_ref, 'ko', markersize=8, label="Reference")

        # Plot models
        for _, row in subset.iterrows():
            theta = np.arccos(np.clip(row["corr"], -1, 1))
            ax.plot(theta, row["std_model"], 'o', label=row["model"])

        ax.set_title(f"{spi}", fontsize=12, weight="bold")
        if spi == "SPI_1":
            ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.1))

    plt.suptitle(f"Taylor Diagrams - Station {station}", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main Loop
# -----------------------------
all_results = []
spi_columns = ["SPI_1", "SPI_3", "SPI_6", "SPI_12"]

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["ds"])

    for model_name in ["TFT", "NBEATS", "NHiTS", "TCN", "LSTM", "WTLSTM", "ExtraTrees", "RandomForest", "SVR"]:
        results_list = forecast(df.copy(), spi_columns, station_name, model_name)
        for result in results_list:
            result.update({"station": station_name, "model": model_name})
            all_results.append(result)

# Save results
metrics_df = pd.DataFrame(all_results)
metrics_df.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)
print("✅ Done! Results saved in:", output_folder)

# Plot Taylor diagrams (4-panel per station)
for st in metrics_df["station"].unique():
    taylor_diagram_panel(metrics_df, st, os.path.join(output_folder, f"taylor_{st}.png"))

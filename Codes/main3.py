
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.metrics import rmse
from darts.models import TFTModel, NBEATSModel, NHiTSModel, TCNModel, RegressionModel, RNNModel
from matplotlib import gridspec

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
output_folder = "./Results/r4"
os.makedirs(output_folder, exist_ok=True)

# SPI groups
SPI_GROUPS = [["SPI_1", "SPI_3"], ["SPI_6", "SPI_9"], ["SPI_12", "SPI_24"]]


def taylor_diagram_panel(metrics_df, station, outfile):
    spi_list = ["SPI_1", "SPI_3", "SPI_6", "SPI_9", "SPI_12", "SPI_24"]

    # Collect subsets & radial max
    subsets, rmax = [], 0.0
    for spi in spi_list:
        sub = metrics_df[(metrics_df["station"] == station) & (metrics_df["spi"] == spi)]
        subsets.append(sub)
        if not sub.empty:
            rmax = max(rmax, float(sub["std_ref"].iloc[0]), float(sub["std_model"].max()))
    if rmax == 0:
        print(f"[Taylor] No data for station {station}; skipping.")
        return
    rmax *= 1.15

    # Use GridSpec for tighter control
    # fig = plt.figure(figsize=(18, 16))
    fig = plt.figure(figsize=(14, 16))

    # gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.05, hspace=0.15)
    gs = fig.add_gridspec(3, 2, wspace=0.05, hspace=0.25)  # 👈 increased hspace

    axes = [fig.add_subplot(gs[i, j], polar=True) for i in range(3) for j in range(2)]

    # correlation grid values
    corrs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    angles = np.arccos(corrs)

    for ax, spi, sub in zip(axes, spi_list, subsets):
        # quarter-circle setup
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_rlim(0, rmax)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_thetagrids(np.degrees(angles), labels=[f"{c:.2f}" for c in corrs], fontsize=8)
        ax.set_rlabel_position(135)

        if sub.empty:
            ax.set_title(f"{spi}\n(no data)", fontsize=11, weight="bold")
            continue

        std_ref = float(sub["std_ref"].iloc[0])
        ax.plot([0], [std_ref], "k*", markersize=10, label="Reference")

        # models
        for _, row in sub.iterrows():
            theta = np.arccos(np.clip(row["corr"], 0, 1))
            ax.plot(theta, row["std_model"], "o", label=row["model"], markersize=6)

        # RMSD contours
        rs, ts = np.meshgrid(
            np.linspace(0, rmax, 200),
            np.linspace(0, np.pi / 2, 200)
        )
        rms = np.sqrt(std_ref**2 + rs**2 - 2 * std_ref * rs * np.cos(ts))
        ax.contour(ts, rs, rms, levels=5, colors="lightgray", linewidths=0.6)

        ax.set_title(f"{spi}", fontsize=11, weight="bold", pad=10)

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), bbox_to_anchor=(0.5, 0.98))

    plt.suptitle(f"Quarter-Circle Taylor Diagrams — Station {station}", fontsize=16, weight="bold", y=0.995)
    plt.subplots_adjust(bottom=0.08)  

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Model training & forecast
# -----------------------------
def train_and_forecast(df, spi, model_name):
    df_spi = df[["ds", spi]].dropna().reset_index(drop=True)

    use_scaler = model_name in ["SVR", "LSTM", "WTLSTM", "TFT", "NBEATS", "NHiTS", "TCN"]
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
    # Train & evaluate
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

    # Forecast till 2099
    last_date = df["ds"].max()
    months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)
    forecast = model.predict(months_to_2099, series=series)

    return {
        "rmse": rmse_val, "corr": corr_val, "std_ref": std_ref,
        "std_model": std_sim, "crmse": crmse_val, "spi": spi,
        "model": model_name, "scaler": scaler, "forecast": forecast, "pred": pred, "series": series
    }

# -----------------------------
# Plot helpers
# -----------------------------
def plot_final_forecasts(station, results, outfile):
    fig, axes = plt.subplots(3, 2, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for i, res in enumerate(results):
        df = res["series"].to_dataframe().reset_index()
        spi = res["spi"]
        scaler = res["scaler"]

        # Historical
        axes[i].plot(df["ds"], df[df.columns[1]], lw=0.6, alpha=0.7, label="Historical")

        # Prediction
        if res["pred"] is not None:
            p = res["pred"].values().flatten()
            if scaler:
                p = scaler.inverse_transform(p.reshape(-1, 1)).flatten()
            axes[i].plot(res["pred"].time_index, p, lw=0.7, color="red", label="Prediction")

        # Forecast
        f = res["forecast"].values().flatten()
        if scaler:
            f = scaler.inverse_transform(f.reshape(-1, 1)).flatten()
        axes[i].plot(res["forecast"].time_index, f, lw=0.7, color="green", label="Forecast")

        axes[i].set_title(f"{spi} — Best: {res['model']}\nRMSE={res['rmse']:.2f}, r={res['corr']:.2f}", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend(fontsize=8)

    plt.suptitle(f"Station {station} — Forecasts till 2099", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

def plot_heatmaps(station, results, outfile):
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    # Gather all SPI values to fix color scale
    all_values = []
    for res in results:
        combined = res["series"].concatenate(res["forecast"])
        all_values.extend(combined.values().flatten())
    vmin, vmax = min(all_values), max(all_values)

    # Plot heatmaps without colorbar first
    for i, res in enumerate(results):
        spi = res["spi"]
        combined = res["series"].concatenate(res["forecast"])
        df = combined.to_dataframe().reset_index()
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df.rename(columns={df.columns[1]: "spi_value"}, inplace=True)

        heatmap_data = df.pivot_table(index="year", columns="month", values="spi_value")

        hm = sns.heatmap(
            heatmap_data,
            cmap="rocket",
            center=0,
            vmin=vmin, vmax=vmax,
            ax=axes[i],
            cbar=False
        )
        axes[i].set_title(f"{spi}", fontsize=12, weight="bold")
        axes[i].set_xlabel("Month")
        axes[i].set_ylabel("Year")
        axes[i].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)

    # Add ONE big colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(hm.collections[0], cax=cbar_ax, label="SPI Value")

    plt.suptitle(f"SPI Heatmaps — Station {station}", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main Loop
# -----------------------------
all_results = []

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["ds"])

    best_results = []

    for group in SPI_GROUPS:
        for spi in group:
            model_metrics = []
            for model_name in ["TFT", "NBEATS", "NHiTS", "TCN", "LSTM", "WTLSTM", "ExtraTrees", "RandomForest", "SVR"]:
                res = train_and_forecast(df.copy(), spi, model_name)
                model_metrics.append(res)
                all_results.append({k: v for k, v in res.items() if k not in ["forecast", "pred", "series", "scaler"]} | {"station": station})

            # Pick best model (lowest RMSE)
            best = min(model_metrics, key=lambda x: x["rmse"])
            best_results.append(best)

    # Save plots per station
    plot_final_forecasts(station, best_results, os.path.join(output_folder, f"{station}_forecasts.png"))
    plot_heatmaps(station, best_results, os.path.join(output_folder, f"{station}_heatmaps.png"))

# Save metrics
metrics_df = pd.DataFrame(all_results)
metrics_df.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)
print("✅ Done! Results saved in:", output_folder)

# Taylor diagrams
for st in metrics_df["station"].unique():
    taylor_diagram_panel(metrics_df, st, os.path.join(output_folder, f"taylor_{st}.png"))


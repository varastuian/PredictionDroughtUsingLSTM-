
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.metrics import mae, rmse, mape
from darts.models import (
    TFTModel, NBEATSModel, NHiTSModel, TCNModel, BlockRNNModel, RegressionModel,
    ARIMA, ExponentialSmoothing
)
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# from matplotlib import gridspec

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import pearsonr
import pywt



def taylor_diagram_panel(metrics_df, station, outfile):

    # Collect subsets & radial max
    subsets, rmax = [], 0.0
    for spi in SPI:
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

    for ax, spi, sub in zip(axes, SPI, subsets):
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
            # if scaler:
            #     pred_inv = scaler.inverse_transform(res["pred"])
            #     p = pred_inv.values().flatten()
            # else:
            p = res["pred"].values().flatten()
            axes[i].plot(res["pred"].time_index, p, lw=0.7, color="red", label="Prediction")

        # Forecast
        # if scaler:
        #     forecast_inv = scaler.inverse_transform(res["forecast"])
        #     f = forecast_inv.values().flatten()
        # else:
        f = res["forecast"].values().flatten()
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
            cbar=False,
            linewidths=0,
            linecolor="none"
            # square=True
            # xticklabels=True,
            # yticklabels=True
        )
        axes[i].set_title(f"{spi}", fontsize=12, weight="bold")
        axes[i].set_xlabel("Month")
        axes[i].set_ylabel("Year")
        axes[i].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)
        # Remove all spines and ticks
        # axes[i].spines[:].set_visible(False)  # remove spines
        # axes[i].tick_params(left=False, bottom=False)  # remove tick marks
    # Add ONE big colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(hm.collections[0], cax=cbar_ax, label="SPI Value")

    plt.suptitle(f"SPI Heatmaps — Station {station}", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Global Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
window_size = 36
horizon = 12
num_epochs = 100
input_folder = "./Data/testdata"
output_folder = "./Results/r19"
os.makedirs(output_folder, exist_ok=True)

# SPI groups
# SPI = ["SPI_1", "SPI_3", "SPI_6", "SPI_9", "SPI_12", "SPI_24"]
SPI = [ "SPI_12"]



def forecast_covariate_to_2099(df, col, last_date):
    """
    df: pandas dataframe with columns ['ds', col]
    last_date: pd.Timestamp of last observed date
    returns: TimeSeries of forecasts starting from first future month (monthly freq)
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds").asfreq("MS")  # monthly start
    
    df = df.reset_index()
    scaler = Scaler()
    series_scaled = scaler.fit_transform(TimeSeries.from_dataframe(df, 'ds', col))
    months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)
    if months_to_2099 <= 0:
        return TimeSeries.from_dataframe(pd.DataFrame({"ds": [], col: []}), "ds", col)  # nothing to do


    model = BlockRNNModel(
            model='LSTM',
            input_chunk_length=window_size,
            output_chunk_length=horizon,
            n_epochs=num_epochs,
            dropout=0.2,
            hidden_dim=64,
            batch_size=16,
            random_state=SEED
        )


    model.fit(series_scaled)
    fc_scaled = model.predict(n=months_to_2099, series=series_scaled)
    fc = scaler.inverse_transform(fc_scaled)
    series = scaler.inverse_transform(series_scaled)

    return series,fc 


def build_future_covariates(df, last_date, window_size=36):
    """
    Build future covariates (temperature & precipitation) to 2099.
    Returns: concatenated past window + future forecasts as a TimeSeries.
    """
    # cov_df = df[["ds", "tm_m", "precip"]].dropna().reset_index(drop=True)
    cov_df = df[["ds", "tm_m", "precip"]]
    hist_tm, fc_tm = forecast_covariate_to_2099(cov_df, "tm_m", last_date)
    hist_pr, fc_pr = forecast_covariate_to_2099(cov_df, "precip", last_date)
    hist_ts = hist_tm.stack(hist_pr)  # combine into multivariate series

    last_cov = hist_ts[-window_size:] if len(hist_ts) >= window_size else hist_ts



    future_df = pd.DataFrame({
        "ds": fc_tm.time_index,
        "tm_m": fc_tm.values().flatten(),
        "precip": fc_pr.values().flatten()
    })
    future_ts = TimeSeries.from_dataframe(future_df, "ds", ["tm_m", "precip"])

    full_future_cov = last_cov.concatenate(future_ts)
    # a,b = hist_ts.split_before(0.95)
    return full_future_cov, hist_ts, future_ts

def plot_covariate_forecasts(hist_ts, future_ts, covariate, color):
    """
    Plot historical vs forecasted covariate.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(hist_ts.time_index, hist_ts[covariate].values().flatten(), label=f"Historical {covariate}", color=color ,lw=0.5)
    plt.plot(future_ts.time_index, future_ts[covariate].values().flatten(), label=f"Forecast {covariate}", color=color, lw=0.5)
    plt.axvline(x=hist_ts.end_time(), color="red", linestyle=":", lw=1.5, label="Forecast Start")
    plt.title(f"{covariate} — Historical vs Forecasted to 2099")
    plt.xlabel("Date")
    plt.ylabel(covariate)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outfile = os.path.join(output_folder, f"covariate {covariate}_{station}.png") 
    plt.savefig(outfile, dpi=300, bbox_inches="tight") 
    plt.close()

# -----------------------------
# Model training & forecast
# -----------------------------
def train_and_forecast(df, value_col, model_name,future_covariates_ts=None):
    """
    
    """
    temp_col = "tm_m"
    precip_col = "precip"
    
    df_spi = df[["ds", value_col, temp_col, precip_col]].dropna().reset_index(drop=True)
    if model_name =="WTLSTM":
        for col in [value_col, "tm_m", "precip"]:
            coeffs = pywt.wavedec(df_spi[col].values, wavelet='db4', level=1)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(df_spi)))
            coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c for i, c in enumerate(coeffs)]
            denoised = pywt.waverec(coeffs_denoised, wavelet="db4")
            df_spi[f"{col}_denoised"] = denoised[:len(df_spi)]
        value_col = f"{value_col}_denoised"
        value_col = f"{temp_col}_denoised"
        value_col = f"{precip_col}_denoised"

    series_full = TimeSeries.from_dataframe(df_spi, 'ds', value_col)
    covariates_full = TimeSeries.from_dataframe(df_spi, 'ds', ["tm_m", "precip"])

    train, test = series_full.split_before(0.8)
    train_cov, test_cov = covariates_full.split_before(0.8)


    


    scaler = Scaler()
    series = scaler.fit_transform(TimeSeries.from_dataframe(df_spi, 'ds', f"{value_col}_denoised"))
    
    cov_scaler = Scaler()
    covariates = cov_scaler.fit_transform(TimeSeries.from_dataframe(df_spi, "ds", ["tm_m_denoised", "precip_denoised"]))


    train, test = series.split_before(0.8)
    train_cov, test_cov = covariates.split_before(0.8)




    use_scaler = model_name in ["SVR", "LSTM", "TFT", "NBEATS", "NHiTS", "TCN"]
    # scaler = cov_scaler=None
    if use_scaler:
        scaler = Scaler()
        # series = scaler.fit_transform(TimeSeries.from_dataframe(df_spi, 'ds', value_col))
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)
        
        cov_scaler = Scaler()
        # covariates = cov_scaler.fit_transform(TimeSeries.from_dataframe(df_spi, "ds", ["tm_m", "precip"]))
        train_cov_scaled = cov_scaler.fit_transform(train_cov)
        test_cov_scaled = cov_scaler.transform(test_cov)
        series = train_scaled  # for training calls
        covariates = train_cov_scaled
    else:
        # scaler = None
        # series = TimeSeries.from_dataframe(df_spi, "ds", value_col)
        # covariates = TimeSeries.from_dataframe(df_spi, "ds", ["tm_m", "precip"])

        scaler = cov_scaler = None
        series = train
        covariates = train_cov


    # -----------------------------
    # Model selection
    # -----------------------------
    if model_name == "ARIMA":
        model = ARIMA()
    elif model_name == "ETS":
        model = ExponentialSmoothing()
    elif model_name == "ExtraTrees":
        model = RegressionModel(ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
                                lags=window_size, output_chunk_length=horizon,lags_past_covariates=window_size)
    elif model_name == "RandomForest":
        model = RegressionModel(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
                                lags=window_size, output_chunk_length=horizon,lags_past_covariates=window_size)
    elif model_name == "SVR":
        model = RegressionModel(SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
                                lags=window_size, output_chunk_length=horizon,lags_past_covariates=window_size)
    elif model_name == "LSTM":
        model = BlockRNNModel(model="LSTM", input_chunk_length=window_size, output_chunk_length=horizon,
                         n_epochs=num_epochs, dropout=0.2,
                         hidden_dim=64, batch_size=16, random_state=SEED)
    elif model_name == "WTLSTM":
        model = BlockRNNModel(model='LSTM', input_chunk_length=window_size, output_chunk_length=horizon,
                         n_epochs=num_epochs, dropout=0.2,
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
    
    if model_name in ["ARIMA", "ETS"]:
        model.fit(train)
    else:
        model.fit(train, past_covariates=train_cov)


    pred = model.predict(n=len(test), series=train, past_covariates=train_cov,future_covariates=test_cov)



    print("train len:", len(train), "test len:", len(test))
    print("train_cov len:", len(train_cov), "test_cov len:", len(test_cov))
    print("train start/end:", train.start_time(), train.end_time())
    print("train_cov start/end:", train_cov.start_time(), train_cov.end_time())


    if use_scaler:
        pred = scaler.inverse_transform(pred)
        test = scaler.inverse_transform(test)
        o = np.array(test.values().flatten())
        p = np.array(pred.values().flatten())
    else:
        o = test.values().flatten()
        p = pred.values().flatten()

    rmse_val = rmse(test, pred)
    corr_val = pearsonr(o, p)[0]
    mae_val = mae(test, pred)
    mape_val = mape(test, pred)

    std_ref = np.std(o, ddof=1)
    std_sim = np.std(p, ddof=1)
    crmse_val = np.sqrt(std_ref**2 + std_sim**2 - 2*std_ref*std_sim*corr_val)

    # Forecast till 2099
    if model_name in ["ARIMA", "ETS"]:
        model.fit(series)
    else:
        model.fit(series, past_covariates=covariates)

    if future_covariates_ts is None:
        last_date = df_spi["ds"].max()
        months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                        end="2099-12-01", freq='MS')


        last_covariates = covariates[-window_size:]
        if model_name == "WTLSTM":
            future_cov = pd.DataFrame({
                'ds': future_dates,
                'tm_m_denoised': np.full(len(future_dates), df_spi['tm_m_denoised'].mean()),  # example: mean temperature
                'precip_denoised': np.full(len(future_dates), df_spi['precip_denoised'].mean())  # example: mean precipitation
            })
            future_cov_ts = TimeSeries.from_dataframe(future_cov, 'ds', ['tm_m_denoised', 'precip_denoised'])

        else:
            future_cov = pd.DataFrame({
                'ds': future_dates,
                'tm_m': np.full(len(future_dates), df_spi['tm_m'].mean()),  # example: mean temperature
                'precip': np.full(len(future_dates), df_spi['precip'].mean())  # example: mean precipitation
                })
            future_cov_ts = TimeSeries.from_dataframe(future_cov, 'ds', ['tm_m', 'precip'])
        full_future_cov = covariates[-window_size:].concatenate(future_cov_ts)

    else:
        full_future_cov = future_covariates_ts

    if use_scaler:
        full_future_cov = cov_scaler.transform(full_future_cov)


    months_to_2099 = len(full_future_cov) - min(window_size, len(covariates))  # future length



    
    if model_name in ["ARIMA", "ETS"]:
            forecast = model.predict(months_to_2099)
    else:
        forecast = model.predict(
        n=months_to_2099,
        series=series,
        past_covariates=full_future_cov
    )

        
        

    plt.figure(figsize=(16,6))
    plt.plot(df['ds'], df[value_col], label="Historical", lw=0.6)
    plt.plot(pred.time_index, p, label="Predicted", lw=0.4, color="red", linestyle="--")
    plt.plot(forecast.time_index, forecast.values(), label="Forecast", lw=0.6, color="green")
    plt.title(f"{value_col} {model_name} Forecast till 2099")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
    plt.legend()
    plt.grid(True)
    metrics_text = f"RMSE: {rmse_val:.3f}\nCorr: {corr_val:.3f}"
    plt.gca().text(
        0.02, 0.95, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
    )
    outfile = os.path.join(output_folder, f"{station}_{value_col}_{model_name}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "spi": spi,"model": model_name,"std_ref": std_ref,
        "std_model": std_sim,"rmse": rmse_val, "corr": corr_val,  "crmse": crmse_val, 
         "scaler": scaler, "forecast": forecast, "pred": pred, "series": series
    }


def pick_best_model(models, weights=None):
    """
    Pick best model using multiple metrics.
    """
    if weights is None:
        weights = {"rmse": 0.5, "crmse": 0.3, "corr": 0.2}

    # Extract metric arrays
    rmse_vals = np.array([m["rmse"] for m in models])
    crmse_vals = np.array([m["crmse"] for m in models])
    corr_vals = np.array([m["corr"] for m in models])

    # Normalize metrics
    rmse_norm = (rmse_vals - rmse_vals.min()) / (rmse_vals.max() - rmse_vals.min() + 1e-8)
    crmse_norm = (crmse_vals - crmse_vals.min()) / (crmse_vals.max() - crmse_vals.min() + 1e-8)
    corr_norm = (corr_vals - corr_vals.min()) / (corr_vals.max() - corr_vals.min() + 1e-8)

    # Compute combined score (lower is better)
    scores = weights["rmse"]*rmse_norm + weights["crmse"]*crmse_norm - weights["corr"]*corr_norm

    best_idx = np.argmin(scores)
    return models[best_idx]
# -----------------------------
# Main Loop
# -----------------------------
all_results = []

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["ds"])
    last_date = df['ds'].max()
    future_covariates_ts, hist_ts, future_ts = build_future_covariates(df, last_date)

    # Plot separately
    plot_covariate_forecasts(hist_ts, future_ts, "tm_m", color="blue")
    plot_covariate_forecasts(hist_ts, future_ts, "precip", color="green")
    # exit()


    best_results = []
    for spi in SPI:
        model_metrics = []
        # for model_name in ["ExtraTrees","RandomForest","SVR","LSTM","WTLSTM","ARIMA","ETS","TFT","NBEATS","NHiTS","TCN"]:
        for model_name in ["WTLSTM","ExtraTrees","RandomForest","SVR","LSTM"]:
            print(f"#______⬇️running :{station} {spi} {model_name}")

            # res = train_and_forecast(df, spi, model_name)
            res = train_and_forecast(df, spi, model_name, future_covariates_ts=future_covariates_ts)

            model_metrics.append(res)
            all_results.append({"station": station}| {k: v for k, v in res.items() if k not in ["forecast", "pred", "series", "scaler"]} )
            print(f"#______end of :{station} {spi} {model_name}")

        # Pick best model (lowest RMSE)
        # best = min(model_metrics, key=lambda x: x["rmse"])
        # best_results.append(best)
        best_results.append(pick_best_model(model_metrics))

    # Save plots per station
    plot_final_forecasts(station, best_results, os.path.join(output_folder, f"forecast_{station}.png"))
    plot_heatmaps(station, best_results, os.path.join(output_folder, f"heatmap_{station}.png"))

# Save metrics
metrics_df = pd.DataFrame(all_results)
metrics_df.to_csv(os.path.join(output_folder, "summary_metrics.csv"), index=False)
print("✅ Done! Results saved in:", output_folder)

# Taylor diagrams
for st in metrics_df["station"].unique():
    taylor_diagram_panel(metrics_df, st, os.path.join(output_folder, f"taylor_{st}.png"))


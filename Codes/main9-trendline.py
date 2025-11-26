import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from darts import TimeSeries
from darts.models import BlockRNNModel, RegressionModel,RandomForest,XGBModel
# from darts.metrics import mae, mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.statistics import check_seasonality , plot_acf
import pywt
from darts.utils.statistics import remove_seasonality


class ForecastConfig:
    """Configuration class for forecasting parameters"""
    def __init__(self):
        self.SEED = 42
        self.horizon =  6
        # self.num_epochs = 350
        self.num_epochs = 50
        self.input_folder = "./Data/python_spi"
        self.output_folder = "./Results/r22"
        self.SPI = ["SPI_1", "SPI_3", "SPI_6", "SPI_9", "SPI_12", "SPI_24"]
        self.models_to_test = ["ExtraTrees", "RandomForest", "SVR", "LSTM","WTLSTM"]
        self.train_test_split = 0.8
        self.lstm_hidden_dim = 64
        # self.lstm_hidden_dim = 1
        self.lstm_dropout = 0.0
        self.lstm_layers = 2
        # self.lstm_layers = 1
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)

        np.random.seed(self.SEED)


def plot_raw_data(df, station, config):
    """Plot raw time series data (SPI, precipitation, temperature)."""
    plt.figure(figsize=(16,8))
    for col in ["precip", "tm_m"]:
        if col in df.columns:
            plt.plot(df["ds"], df[col], lw=0.7, label=col)
    plt.title(f"Raw Data — Station {station}")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"rawdata_{station}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter(observed, predicted, station, spi, model, config):
    plt.figure(figsize=(6,6))
    plt.scatter(observed, predicted, alpha=0.5, edgecolor="k")
    lims = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    plt.plot(lims, lims, "r--", lw=1.5, label="1:1 Line")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{station} - {spi} ({model})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"scatter_{station}_{spi}_{model}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_distribution(observed, predicted, station, spi, model, config):
    residuals = observed - predicted
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"Residuals — {station} {spi} ({model})")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    outfile = os.path.join(config.output_folder, f"residuals_{station}_{spi}_{model}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_error(observed, predicted, time_index, station, spi, model, config, window=12):
    errors = (observed - predicted)**2
    rolling_rmse = np.sqrt(pd.Series(errors, index=time_index).rolling(window).mean())
    plt.figure(figsize=(12,5))
    rolling_rmse.plot(color="blue", lw=1.5)
    plt.title(f"Rolling RMSE ({window}-month) — {station} {spi} ({model})")
    plt.ylabel("RMSE")
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"rollingrmse_{station}_{spi}_{model}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_seasonal_cycle(hist_ts, forecast_ts, station, spi, config):
    df_hist = hist_ts.pd_dataframe().reset_index()
    df_fore = forecast_ts.pd_dataframe().reset_index()
    df_hist["month"] = df_hist["ds"].dt.month
    df_fore["month"] = df_fore["ds"].dt.month
    
    monthly_hist = df_hist.groupby("month").mean()
    monthly_fore = df_fore.groupby("month").mean()
    
    plt.figure(figsize=(10,5))
    plt.plot(monthly_hist.index, monthly_hist[spi], marker="o", label="Observed")
    plt.plot(monthly_fore.index, monthly_fore[spi], marker="s", label="Forecast")
    plt.xticks(range(1,13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.title(f"Seasonal Cycle — {station} {spi}")
    plt.ylabel(spi)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"seasonalcycle_{station}_{spi}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_boxplots(metrics_df, config):
    metrics = ["rmse", "mae", "corr", "mape"]
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.boxplot(x="spi", y=metric, hue="model", data=metrics_df)
        plt.title(f"Model Comparison by {metric.upper()}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        outfile = os.path.join(config.output_folder, f"boxplot_{metric}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()


def plot_model_ranking(metrics_df, config):
    best_models = metrics_df.groupby(["station", "spi"]).apply(
        lambda g: g.loc[g["rmse"].idxmin(), "model"]
    )
    counts = best_models.value_counts()
    plt.figure(figsize=(8,5))
    counts.plot(kind="bar", color="skyblue", edgecolor="k")
    plt.title("Best Model Counts (Lowest RMSE)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, "bestmodel_counts.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_covariate_forecasts(hist_ts, future_ts, covariate: str, config: ForecastConfig, color: str = "blue"):

    plt.figure(figsize=(14, 6))
    plt.plot(hist_ts.time_index, hist_ts.values().flatten(), label=f"Historical {covariate}", color=color, lw=0.5)
    plt.plot(future_ts.time_index, future_ts.values().flatten(), label=f"Forecast {covariate}", color=color, lw=0.5)
    plt.axvline(x=hist_ts.end_time(), color="red", linestyle=":", lw=1.5, label="Forecast Start")
    plt.title(f"{config.station} - {covariate} - Historical vs Forecasted to 2099")
    plt.xlabel("Date")
    plt.ylabel(covariate)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outfile = os.path.join(config.output_folder, f"covariate_{covariate}_{config.station}.png") 
    plt.savefig(outfile, dpi=300, bbox_inches="tight") 
    plt.close()

def plot_scatter(observed, predicted, station, spi, model, config):
    plt.figure(figsize=(6,6))
    plt.scatter(observed, predicted, alpha=0.5, edgecolor="k")
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="1:1 Line")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{station} - {spi} ({model})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"scatter_{station}_{spi}_{model}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

def plot_metric_boxplots(metrics_df, config):
    metrics = ["rmse", "mae", "corr", "mape"]
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.boxplot(x="spi", y=metric, hue="model", data=metrics_df)
        plt.title(f"Model Comparison by {metric.upper()}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        outfile = os.path.join(config.output_folder, f"boxplot_{metric}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()


def plot_residual_distribution(observed, predicted, station, spi, model, config):
    residuals = observed - predicted
    plt.figure(figsize=(8,5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"Residual Distribution — {station} {spi} ({model})")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    outfile = os.path.join(config.output_folder, f"residuals_{station}_{spi}_{model}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

def plot_rolling_error(observed, predicted, time_index, station, spi, model, config, window=12):
    errors = (observed - predicted)**2
    rolling_rmse = np.sqrt(pd.Series(errors, index=time_index).rolling(window).mean())
    plt.figure(figsize=(12,5))
    rolling_rmse.plot(color="blue", lw=1.5)
    plt.title(f"Rolling RMSE ({window}-month) — {station} {spi} ({model})")
    plt.ylabel("RMSE")
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"rolling_rmse_{station}_{spi}_{model}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_seasonal_cycle(hist_ts, forecast_ts, station, spi, config):
    df_hist = hist_ts.pd_dataframe().reset_index()
    df_fore = forecast_ts.pd_dataframe().reset_index()
    df_hist["month"] = df_hist["ds"].dt.month
    df_fore["month"] = df_fore["ds"].dt.month
    
    monthly_hist = df_hist.groupby("month").mean()
    monthly_fore = df_fore.groupby("month").mean()
    
    plt.figure(figsize=(10,5))
    plt.plot(monthly_hist.index, monthly_hist[spi], marker="o", label="Observed")
    plt.plot(monthly_fore.index, monthly_fore[spi], marker="s", label="Forecast")
    plt.xticks(range(1,13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.title(f"Seasonal Cycle Comparison — {station} {spi}")
    plt.ylabel(spi)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, f"seasonal_cycle_{station}_{spi}.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_ranking(metrics_df, config):
    best_models = metrics_df.groupby(["station", "spi"]).apply(
        lambda g: g.loc[g["rmse"].idxmin(), "model"]
    )
    counts = best_models.value_counts()
    plt.figure(figsize=(8,5))
    counts.plot(kind="bar", color="skyblue", edgecolor="k")
    plt.title("Best Model Counts (Lowest RMSE)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    outfile = os.path.join(config.output_folder, "best_model_counts.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

def taylor_diagram_panel(config,metrics_df, station, outfile):

    # Collect subsets & radial max
    subsets, rmax = [], 0.0
    for spi in config.SPI:
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

    for ax, spi, sub in zip(axes, config.SPI, subsets):
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
        forecast = res["forecast"]
        # Historical
        axes[i].plot(df["ds"], df[df.columns[1]], lw=0.6, alpha=0.7, label="Historical")

        p = res["pred"].values().flatten()

        axes[i].plot(res["pred"].time_index, p, lw=0.7, color="red", label="Prediction")

        # Forecast
        
        f = forecast.values().flatten()
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
    # plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def build_cyclic_covariates(time_index: pd.DatetimeIndex) -> TimeSeries:

    # month_cov = datetime_attribute_timeseries(time_index, "month", one_hot=True)
    month_cov = datetime_attribute_timeseries(time_index, "month")
    year_cov = datetime_attribute_timeseries(time_index, "year")
    
    year_scaled = (year_cov.values() - year_cov.values().min()) / (year_cov.values().max() - year_cov.values().min())
    year_scaled_ts = TimeSeries.from_times_and_values(time_index, year_scaled)
    
    # cyc_cov = month_cov.stack(year_cov).stack(year_scaled_ts)
    cyc_cov = month_cov
    return cyc_cov

def wavelet_denoise(series: np.ndarray, wavelet: str = "db4", level: int = 2) -> np.ndarray:
    
    coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
    # Universal threshold based on noise estimate
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(series)))
    coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c for i, c in enumerate(coeffs)]
    denoised = pywt.waverec(coeffs_denoised, wavelet=wavelet)
    return denoised[:len(series)]

def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:

    std_ref = np.std(observed, ddof=1)
    std_sim = np.std(predicted, ddof=1)
    corr_val = pearsonr(observed, predicted)[0]
    rmse_val = np.sqrt(mean_squared_error(observed, predicted))
    mae_val = mean_absolute_error(observed, predicted)
    mape_val = np.mean(np.abs((observed - predicted) / observed)) * 100
    
    # Avoid division by zero in CRMSLE calculation
    if std_ref > 0 and std_sim > 0:
        crmse_val = np.sqrt(std_ref**2 + std_sim**2 - 2 * std_ref * std_sim * corr_val)
    else:
        crmse_val = np.nan
        print("Cannot calculate CRMSE due to zero standard deviation")
    
    return {
        "std_ref": std_ref,
        "std_model": std_sim,
        "rmse": rmse_val,
        "corr": corr_val,
        "crmse": crmse_val,
        "mae": mae_val,
        "mape": mape_val
    }

def forecast_covariate_to_2099(df: pd.DataFrame, col: str, config: ForecastConfig) -> Tuple[TimeSeries, TimeSeries]:
    
    df = df.copy()

    series = TimeSeries.from_dataframe(df, 'ds', col)
    window_size = 12

    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)


    cyc_cov = build_cyclic_covariates(series.time_index)
    cov_scaler = Scaler()
    cyc_cov = cov_scaler.fit_transform(cyc_cov)

    model = BlockRNNModel(
        model='LSTM',
        input_chunk_length=window_size,
        output_chunk_length=config.horizon,
       n_epochs=config.num_epochs,
        dropout=config.lstm_dropout,
        hidden_dim=config.lstm_hidden_dim,
        n_rnn_layers=config.lstm_layers,

        random_state=config.SEED
    )

    model.fit(series_scaled
            #   , past_covariates=cyc_cov
              )



    # Create future covariates for prediction
    future_time_idx = pd.date_range(
        start=series.time_index[0], 
        periods=len(series) + config.months_to_2099, 
        freq="MS"
    )

    cyc_cov_future = build_cyclic_covariates(future_time_idx)

    cyc_cov_future_scaled = cov_scaler.transform(cyc_cov_future)

    # Predict
    fc_scaled = model.predict(n=config.months_to_2099, series=series_scaled
                            #   , past_covariates=cyc_cov_future_scaled
                              )
    fc = scaler.inverse_transform(fc_scaled)

    # Clip negative precipitation to 0
    if col == "precip":
        fc = fc.map(lambda x: np.clip(x, 0, None))

    return series, fc

def forecast_precip_to_2099(df: pd.DataFrame, config: ForecastConfig) -> Tuple[TimeSeries, TimeSeries]:
    df = df.copy()

    # Stage 1: Wet/Dry
    df['wet_day'] = (df['precip'] > 0).astype(int)
    hist_wet, fc_wet = forecast_covariate_to_2099(df,  "wet_day", config)

    # Stage 2: Positive precipitation (log1p)
    df['log_precip'] = np.log1p(df['precip'])  # keep all months
    hist_log, fc_log = forecast_covariate_to_2099(df, "log_precip", config)

    # Convert back
    fc_intensity = TimeSeries.from_times_and_values(fc_log.time_index, np.expm1(fc_log.values().flatten()))

    # Align time indices
    future_idx = fc_wet.time_index
    wet_flag = (fc_wet.values().flatten() > 0.5).astype(float)
    fc_precip_final = TimeSeries.from_times_and_values(future_idx, wet_flag * fc_intensity.values().flatten())

    hist_precip = TimeSeries.from_dataframe(df, "ds", ["precip"])
    return hist_precip, fc_precip_final

def build_future_covariates(df: pd.DataFrame, config: ForecastConfig) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:

    # Forecast temperature and precipitation
    hist_pr, fc_pr = forecast_covariate_to_2099(df,"precip", config)
    # hist_pr, fc_pr = forecast_precip_to_2099(df, config)
    plot_covariate_forecasts(hist_pr, fc_pr, "precip", config, color="green")


    hist_tm, fc_tm = forecast_covariate_to_2099(df, "tm_m", config)
    plot_covariate_forecasts(hist_tm, fc_tm, "tm_m", config, color="blue")
        
        
    # Combine historical covariates
    hist_cov = hist_tm.stack(hist_pr)

    # Create future covariates DataFrame
    future_df = pd.DataFrame({
        "ds": fc_tm.time_index,
        "tm_m": fc_tm.values().flatten(),
        "precip": fc_pr.values().flatten()
    })
    future_cov = TimeSeries.from_dataframe(future_df, "ds", ["tm_m", "precip"])

    # Combine historical and future covariates
    full_cov = hist_cov.concatenate(future_cov)
    cyc_cov = build_cyclic_covariates(full_cov.time_index)


    # hist_cov = hist_cov.stack(cyc_cov.split_before(future_cov.start_time())[0])
    # full_cov = full_cov.stack(cyc_cov)
    

    return full_cov, hist_cov

def prepare_wavelet_data(value_col,train, test,hist, train_cov, test_cov,hist_cov,full_cov):
   
    # Convert to DataFrames
    train_df = train.to_dataframe().reset_index()
    test_df = test.to_dataframe().reset_index()
    hist_df = hist.to_dataframe().reset_index()

    traincov_df = train_cov.to_dataframe().reset_index()
    testcov_df = test_cov.to_dataframe().reset_index()
    hist_cov_df = hist_cov.to_dataframe().reset_index()
    full_cov_df = full_cov.to_dataframe().reset_index()

    # Apply wavelet denoising
    train_df[f"{value_col}_denoised"] = wavelet_denoise(train_df[value_col].values)
    test_df[f"{value_col}_denoised"] = wavelet_denoise(test_df[value_col].values)
    hist_df[f"{value_col}_denoised"] = wavelet_denoise(hist_df[value_col].values)

    traincov_df["tm_m_denoised"] = wavelet_denoise(traincov_df["tm_m"].values)
    testcov_df["tm_m_denoised"] = wavelet_denoise(testcov_df["tm_m"].values)
    hist_cov_df["tm_m_denoised"] = wavelet_denoise(hist_cov_df["tm_m"].values)
    full_cov_df["tm_m_denoised"] = wavelet_denoise(full_cov_df["tm_m"].values)

    traincov_df["precip_denoised"] = wavelet_denoise(traincov_df["precip"].values)
    testcov_df["precip_denoised"] = wavelet_denoise(testcov_df["precip"].values)
    hist_cov_df["precip_denoised"] = wavelet_denoise(hist_cov_df["precip"].values)
    full_cov_df["precip_denoised"] = wavelet_denoise(full_cov_df["precip"].values)

    # Get cyclic columns
    cyclic_cols = [col for col in full_cov_df.columns if "month" in col or "year" in col]

    # Rebuild TimeSeries objects
    train_denoised = TimeSeries.from_dataframe(train_df, "ds", f"{value_col}_denoised")
    test_denoised = TimeSeries.from_dataframe(test_df, "ds", f"{value_col}_denoised")
    hist_denoised = TimeSeries.from_dataframe(hist_df, "ds", f"{value_col}_denoised")


    train_cov_denoised = TimeSeries.from_dataframe(traincov_df, "ds", ["tm_m_denoised", "precip_denoised"] + cyclic_cols)
    test_cov_denoised = TimeSeries.from_dataframe(testcov_df, "ds", ["tm_m_denoised", "precip_denoised"] + cyclic_cols)
    hist_cov_denoised = TimeSeries.from_dataframe(hist_cov_df, "ds", ["tm_m_denoised", "precip_denoised"] + cyclic_cols)
    full_cov_denoised = TimeSeries.from_dataframe(full_cov_df, "ds", ["tm_m_denoised", "precip_denoised"] + cyclic_cols)

    return train_denoised, test_denoised,hist_denoised, train_cov_denoised, test_cov_denoised,hist_cov_denoised,full_cov_denoised

def create_model(model_name: str, window_size: int, config: ForecastConfig):
    
    if model_name == "ExtraTrees":
        return XGBModel(
            lags=window_size,
            output_chunk_length=config.horizon              
            ,lags_past_covariates=[-i for i in range(1,13)],
        )
    elif model_name == "RandomForest":
        return RandomForest(
            n_estimators=100,random_state=config.SEED,
            lags=window_size, 
            output_chunk_length=config.horizon                      
            ,lags_past_covariates=[-i for i in range(1,13)]
        )
    elif model_name == "SVR":
        return RegressionModel(
            model=SVR(kernel="rbf", C=1, gamma=0.01, epsilon=0.01),
            lags=window_size, 
            output_chunk_length=config.horizon           
            ,lags_past_covariates=[-i for i in range(1,13)]
        )
    elif model_name in ["LSTM","WTLSTM"] :
        return BlockRNNModel(
            model="LSTM", 
            input_chunk_length=window_size, 
            output_chunk_length=config.horizon,
            n_epochs=config.num_epochs, 
            dropout=config.lstm_dropout,
            n_rnn_layers=config.lstm_layers,
            hidden_dim=config.lstm_hidden_dim, 
            random_state=config.SEED
            # ,likelihood=GaussianLikelihood()
        )
    
def pick_best_model(models: List[Dict], weights: Dict[str, float] = None) -> Dict:
    if weights is None:
        weights = {"rmse": 0.5, "crmse": 0.3, "corr": 0.2}

    # Filter out None results
    valid_models = [m for m in models if m is not None]
    if not valid_models:
        return None

    # Extract metric arrays
    rmse_vals = np.array([m["rmse"] for m in valid_models])
    crmse_vals = np.array([m["crmse"] for m in valid_models])
    corr_vals = np.array([m["corr"] for m in valid_models])

    # Normalize metrics
    rmse_norm = (rmse_vals - rmse_vals.min()) / (rmse_vals.max() - rmse_vals.min() + 1e-8)
    crmse_norm = (crmse_vals - crmse_vals.min()) / (crmse_vals.max() - crmse_vals.min() + 1e-8)
    corr_norm = (corr_vals - corr_vals.min()) / (corr_vals.max() - corr_vals.min() + 1e-8)

    # Compute combined score (lower is better)
    scores = (weights["rmse"] * rmse_norm + 
              weights["crmse"] * crmse_norm - 
              weights["corr"] * corr_norm)

    best_idx = np.argmin(scores)
    return valid_models[best_idx]



def main():
    config = ForecastConfig()
    all_results = []
    
    data_files = glob.glob(os.path.join(config.input_folder, "*.csv"))
    
    
    for file in data_files:
        station = os.path.splitext(os.path.basename(file))[0]
        # if station != "40700":
        #     continue
        print(f"Processing station: {station}")
        
        df = pd.read_csv(file, parse_dates=["ds"])
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.set_index("ds").asfreq("MS").reset_index()
        # plot_raw_data(df, station, config)

        last_date = df['ds'].max()
        config.months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month+1)
        config.station = station

        full_cov, hist_cov = build_future_covariates(df, config)

        raw_hist_cov = hist_cov.copy()
        raw_full_cov = full_cov.copy()


        best_results = []
        for spi in config.SPI:
            # if spi != "SPI_6":
            #     continue
            model_metrics = []

            # Prepare data
            df_spi = df[["ds", spi]].dropna().reset_index(drop=True)
            hist = TimeSeries.from_dataframe(df_spi, 'ds', spi)
            

            window_size = 12


            raw_hist = hist.copy()


            for model_name in config.models_to_test:
                # if model_name != "LSTM":
                #     continue
                print(f"Running: {station} {spi} {model_name}")
                            
                # Split data
                hist = raw_hist
                hist_cov = raw_hist_cov.slice_intersect(hist)
                full_cov = raw_full_cov.slice(hist.start_time(), full_cov.end_time())
                train, test = hist.split_before(config.train_test_split)
                train_cov, test_cov = hist_cov.split_before(config.train_test_split)
                test_raw = test.copy()
                if model_name == "WTLSTM":
                    train, test,hist, train_cov, test_cov,hist_cov,full_cov = prepare_wavelet_data(spi,train, test,hist, train_cov, test_cov,hist_cov,full_cov)

                # Scale data if needed
                use_scaler = model_name in ["SVR", "LSTM", "WTLSTM"]
                scaler = None
                
                if use_scaler:
                    scaler = Scaler()
                    train_scaled = scaler.fit_transform(train)
                    test_scaled = scaler.transform(test)
                    hist_scaled = scaler.transform(hist)
                                        
                    cov_scaler = Scaler()
                    train_cov_scaled = cov_scaler.fit_transform(train_cov)
                    # test_cov_scaled = cov_scaler.transform(test_cov)
                    hist_cov_scaled = cov_scaler.transform(hist_cov)
                    full_cov_scaled = cov_scaler.transform(full_cov)
                else:
                    train_scaled = train
                    # test_scaled = test
                    hist_scaled = hist

                    train_cov_scaled = train_cov
                    # test_cov_scaled = test_cov
                    hist_cov_scaled = hist_cov
                    full_cov_scaled =full_cov


                # Create and train model
                model = create_model(model_name, window_size, config)
                
                
                model.fit(train_scaled
                          , past_covariates=train_cov_scaled
                          )

                pred = model.predict(n=len(test), series=train_scaled
                                     , past_covariates=hist_cov_scaled
                                     )
             
                # Inverse transform if scaled
                if use_scaler:
                    pred = scaler.inverse_transform(pred)
                    test = scaler.inverse_transform(test)

                # Calculate metrics
                observed = test_raw.values().flatten()
                predicted = pred.values().flatten()
                metrics = calculate_metrics(observed, predicted)


                plot_scatter(observed, predicted, station, spi, model_name, config)
                # plot_residual_distribution(observed, predicted, station, spi, model_name, config)
                # plot_rolling_error(observed, predicted, test_raw.time_index, station, spi, model_name, config)
                
                #==========================================
                # Refit model on full historical data
                #==========================================
                model.fit(hist_scaled
                , past_covariates=hist_cov_scaled
                )

                
                # Make forecast
                forecast = model.predict(
                        n=config.months_to_2099,
                        series=hist_scaled
                        ,past_covariates=full_cov_scaled
                )
                
                if scaler is not None:
                    forecast = scaler.inverse_transform(forecast)
                    hist_scaled = scaler.inverse_transform(hist_scaled)

                # Plot results
                plt.figure(figsize=(16, 6))
                plt.plot(hist.time_index, hist.values(), label="Historical", lw=0.6)
                plt.plot(pred.time_index, predicted, label="Predicted", lw=0.4, color="red", linestyle="--")
                plt.plot(forecast.time_index, forecast.values(), label="Forecast", lw=0.6, color="green")
                plt.title(f"{station} {spi} {model_name} Forecast till 2099")
                plt.xlabel("Date")
                plt.ylabel(spi)
                plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
                plt.legend()
                plt.grid(True)
                
                # Add metrics text box
                metrics_text = f"RMSE: {metrics['rmse']:.3f}\nCorr: {metrics['corr']:.3f}"
                plt.gca().text(
                    0.02, 0.95, metrics_text,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
                )

                # Add trend line for forecast
                forecast_df = forecast.to_dataframe()
                x = np.arange(len(forecast_df))
                y = forecast_df.iloc[:, 0].values
            

                # # Fit linear trend
                # coef = np.polyfit(x, y, 1)
                # trend = np.polyval(coef, x)

                # # Plot trend line
                # plt.plot(forecast_df.index, trend, label="Trend Line", linestyle="--")
                dates = forecast_df.index

                # ---------------------------------------------------
                # 1) GLOBAL LINEAR TREND (whole forecast)
                # ---------------------------------------------------
                coef = np.polyfit(x, y, 1)
                global_trend = np.polyval(coef, x)

                plt.plot(
                    dates,
                    global_trend,
                    label="Global Trend",
                    linestyle="--",
                    linewidth=2,
                    color="blue",
                )

                # ---------------------------------------------------
                # 2) DECADE TREND LINES
                # ---------------------------------------------------
                decade_length = 120  # 120 months = 10 years
                n = len(y)

                decade_start_idx = list(range(0, n, decade_length))

                for start in decade_start_idx:
                    end = min(start + decade_length, n)

                    # Extract decade slice
                    x_dec = x[start:end]
                    y_dec = y[start:end]
                    date_dec = dates[start:end]

                    # Fit trend line for this decade
                    coef_dec = np.polyfit(x_dec, y_dec, 1)
                    trend_dec = np.polyval(coef_dec, x_dec)

                    # Plot decade trend line
                    plt.plot(
                        date_dec,
                        trend_dec,
                        linestyle="-",
                        linewidth=2,
                        alpha=0.9,
                        label=f"Decade Trend {date_dec[0].year}-{date_dec[-1].year}",
                    )

                
                outfile = os.path.join(config.output_folder, f"{station} {spi}_{model_name}.png")
                plt.savefig(outfile, dpi=300, bbox_inches="tight")
                plt.close() 
                res = {
                    "spi": spi,
                    "model": model_name,
                    **metrics,
                    "horizon": config.horizon,
                    "window_size": window_size,
                    "epoch": config.num_epochs,
                    "forecast": forecast,
                    "pred": pred,
                    "series": hist
                }

                
                model_metrics.append(res)
                all_results.append({
                    "station": station,
                    **{k: v for k, v in res.items() if k not in ["forecast", "pred", "series", "scaler"]}
                })
            
            # Select best model for this SPI
            best_model = pick_best_model(model_metrics)
            if best_model:
                best_results.append(best_model)
                # plot_seasonal_cycle(best_model["series"], best_model["forecast"], station, best_model["spi"], config)


        # Save plots for this station
        # if best_results:
        #     plot_final_forecasts(station, best_results, os.path.join(config.output_folder, f"forecast_{station}.png"))
        #     plot_heatmaps(station, best_results, os.path.join(config.output_folder, f"heatmap_{station}.png"))
            
       
    
    # # Save metrics and create Taylor diagrams
    # if all_results:
    #     metrics_df = pd.DataFrame(all_results)
    #     metrics_df.to_csv(os.path.join(config.output_folder, "summary_metrics.csv"), index=False)
        
    #     for station in metrics_df["station"].unique():
    #         taylor_diagram_panel(config,metrics_df, station, os.path.join(config.output_folder, f"taylor_{station}.png"))
    #     plot_metric_boxplots(metrics_df, config)
    #     plot_model_ranking(metrics_df, config)

    
    print(f"✅ Done! Results saved in: {config.output_folder}")

if __name__ == "__main__":
    main()
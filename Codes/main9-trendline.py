import datetime
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
from darts.models import BlockRNNModel, RegressionModel,RandomForest,XGBModel,RNNModel,NHiTSModel, TFTModel
# from darts.metrics import mae, mape, rmse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.statistics import check_seasonality , plot_acf
import pywt
from darts.utils.statistics import remove_seasonality




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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()


def plot_seasonal_cycle(hist_ts, forecast_ts, station, spi, config):
    df_hist = hist_ts.to_dataframe().reset_index()
    df_fore = forecast_ts.to_dataframe().reset_index()
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
        plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()


def plot_covariate_forecasts(hist_ts, future_ts, covariate: str, config, color: str = "blue"):

    plt.figure(figsize=(14, 6))
    plt.plot(hist_ts.time_index, hist_ts.values().flatten(), label=f"Historical {covariate}", lw=0.5)
    plt.plot(future_ts.time_index, future_ts.values().flatten(), label=f"Forecast {covariate}", color="green", lw=0.5)
    plt.axvline(x=hist_ts.end_time(), color="red", linestyle=":", lw=1.5, label="Forecast Start")
    plt.title(f"{config.station} - {covariate} ")
    plt.xlabel("Date")
    plt.ylabel(covariate)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outfile = os.path.join(config.output_folder, f"covariate_{covariate}_{config.station}.png") 
    plt.savefig(outfile, dpi=600, bbox_inches="tight") 
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
        plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()


def plot_seasonal_cycle(hist_ts, forecast_ts, station, spi, config):
    df_hist = hist_ts.to_dataframe().reset_index()
    df_fore = forecast_ts.to_dataframe().reset_index()
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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

    plt.suptitle(f"Taylor Diagram — Station {station}", fontsize=16, weight="bold", y=0.995)
    plt.subplots_adjust(bottom=0.08)  

    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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

        axes[i].set_title(f"{spi} — Best: {res['model']}\nRMSE={res['rmse']:.2f}, r={res['corr'][0]:.2f}", fontsize=10)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend(fontsize=8)

        # ----------------------- Global trend line -----------------------
        forecast_df = forecast.to_dataframe()
        x = np.arange(len(forecast_df))
        y = forecast_df.iloc[:, 0].values
        dates = forecast_df.index

        coef = np.polyfit(x, y, 1)
        global_trend = np.polyval(coef, x)
        m_global, b_global = coef

        axes[i].plot(
            dates,
            global_trend,
            label="Global Trend",
            linestyle="--",
            linewidth=2,
            color="blue",
        )

        # --- Place trend equation bottom-right near the line ---
        y_last = global_trend[-1]

        # Offsets based on visible range
        x_offset = (dates[-1] - dates[0]) * 0.03
        y_offset = (max(global_trend) - min(global_trend)) * -0.05

        equation = f"y = {m_global:.4f}x + {b_global:.4f}"

        axes[i].annotate(
            equation,
            xy=(dates[-1], y_last),
            xytext=(dates[-1] - x_offset, y_last + y_offset),
            fontsize=10,
            color="blue",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        # ----------------------- Decade trend lines -----------------------
        decade_length = 240  # 10 years
        n = len(y)

        for start in range(0, n, decade_length):
            end = min(start + decade_length, n)
            x_dec = x[start:end]
            y_dec = y[start:end]
            date_dec = dates[start:end]

            coef_dec = np.polyfit(x_dec, y_dec, 1)
            trend_dec = np.polyval(coef_dec, x_dec)

            axes[i].plot(
                date_dec,
                trend_dec,
                linestyle="-",
                linewidth=2,
                alpha=0.9,
                # label=f"Decade Trend {date_dec[0].year}-{date_dec[-1].year}"
            )

        axes[i].legend()


    plt.suptitle(f"Station {station}", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
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
    plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()


def pick_best_model(models: List[Dict], weights: Dict[str, float] = None) -> Dict:
    if weights is None:
        weights = {"rmse": 0.5, "crmse": 0.3, "corr": 0.2}

    # Filter out None results
    valid_models = [m for m in models if m is not None]
    if not valid_models:
        return None

    # Extract metric arrays
    # rmse_vals = np.array([m["rmse"] for m in valid_models])
    # crmse_vals = np.array([m["crmse"] for m in valid_models])
    # corr_vals = np.array([m["corr"] for m in valid_models])
    print(m for m in valid_models)
    rmse_vals = np.array([float(m["rmse"]) for m in valid_models])
    crmse_vals = np.array([float(m["crmse"]) for m in valid_models])
    corr_vals = np.array([float(m["corr"]) for m in valid_models])


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

def apply_reverse_wavelet(pred_ts, wavelet_info):
    wavelet = wavelet_info["wavelet"]
    coeffs_orig = wavelet_info["train_coeffs"]

    # replace only the "approximation" part with prediction
    coeffs_new = coeffs_orig.copy()
    coeffs_new[0] = pred_ts.values().flatten()

    recon = pywt.waverec(coeffs_new, wavelet)
    recon = recon[:len(pred_ts)]

    return TimeSeries.from_times_and_values(pred_ts.time_index, recon)

# def wavelet_denoise(series: np.ndarray, wavelet: str = "db4", level: int = 1) -> np.ndarray:
    
#     coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
#     # Universal threshold based on noise estimate
#     threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(series)))
#     coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c for i, c in enumerate(coeffs)]
#     denoised = pywt.waverec(coeffs_denoised, wavelet=wavelet)
#     return denoised[:len(series)]

def wavelet_denoise(series, wavelet="db4", level=1):
    coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(series)))
    coeffs_denoised = [coeffs[0]] + [pywt.threshold(c, threshold, "soft") for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs_denoised, wavelet=wavelet)
    return denoised[:len(series)], coeffs  # return coeffs for reconstruction

def prepare_wavelet_data(train, full_cov, return_info=False):

    value_col = train.components[0]
    train_df = train.to_dataframe().reset_index()
    fcov_df = full_cov.to_dataframe().reset_index()

    # --- apply denoising and store original coeffs for reconstruction ---
    train_df[f"{value_col}_denoised"], train_coeffs = wavelet_denoise(train_df[value_col].values)
    fcov_df["tm_m_denoised"], tm_coeffs = wavelet_denoise(fcov_df["tm_m"].values)
    fcov_df["precip_denoised"], pr_coeffs = wavelet_denoise(fcov_df["precip"].values)

    cyclic = [c for c in fcov_df.columns if "month" in c or "year" in c]

    train_d = TimeSeries.from_dataframe(train_df, "ds", f"{value_col}_denoised")
    full_cov_d = TimeSeries.from_dataframe(fcov_df, "ds",
                                           ["tm_m_denoised", "precip_denoised"] + cyclic)

    if return_info:
        return train_d, full_cov_d, {
            "train_coeffs": train_coeffs,
            "wavelet": "db4"
        }

    return train_d, full_cov_d

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


def build_cyclic_covariates(time_index: pd.DatetimeIndex) -> TimeSeries:

    # year as numeric attribute then scaled
    year_ts = datetime_attribute_timeseries(time_index, attribute="year").astype(float)
    year_ts = Scaler().fit_transform(year_ts)

    # month as numeric attribute (1..12) -> convert to sin/cos
    month_ts = datetime_attribute_timeseries(time_index, attribute="month").astype(float)
    df_month = month_ts.to_dataframe(copy=True)
    df_month["month_sin"] = np.sin(2 * np.pi * df_month["month"] / 12.0)
    df_month["month_cos"] = np.cos(2 * np.pi * df_month["month"] / 12.0)
    df_month = df_month[["month_sin", "month_cos"]]

    month_sin_cos_ts = TimeSeries.from_dataframe(df_month)

    # covariates = year_ts.stack(month_sin_cos_ts)
    covariates = month_sin_cos_ts
    return covariates

# -------------------------------
# Forecast single covariate to 2099
# -------------------------------
def forecast_covariate_to_2099(df: pd.DataFrame, col: str, config):
    
    df_local = df.copy().sort_values("ds").reset_index(drop=True)

    series = TimeSeries.from_dataframe(df_local, 'ds', col)

    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)


    full_time_idx = pd.date_range(
        start=series.time_index[0], 
        periods=len(series) + config.months_to_2099, 
        freq="MS"
    )

    cyc_cov = build_cyclic_covariates(full_time_idx)

    split_point = int(len(series_scaled) * config.train_test_split)
    train_end = series_scaled.time_index[split_point - 1]
    train_series, val_series = series_scaled.split_before(train_end + pd.Timedelta(days=1))


    cyc_cov_train = cyc_cov.slice(series_scaled.start_time(), train_series.end_time())
    cyc_cov_val = cyc_cov.slice(val_series.start_time(), val_series.end_time())

    
    model = RNNModel(
        model='LSTM',
        input_chunk_length=config.window_size,
        # output_chunk_length=config.horizon,
        n_epochs=config.num_epochs,
        optimizer_kwargs={"lr": 1e-3},
        training_length=20,
        force_reset=True,
        batch_size=16,
        dropout=config.lstm_dropout,
        hidden_dim=config.lstm_hidden_dim,
        n_rnn_layers=config.lstm_layers,
        random_state=config.SEED,
    # pl_trainer_kwargs={
    #     "callbacks": [EarlyStopping(
    #     monitor="val_loss",
    #     patience=10,
    #     mode="min"
    # )]
    # }
    )

    model.fit(series_scaled
              , future_covariates=cyc_cov
              )

    # model.fit(
    #     series=train_series,
    #     future_covariates=cyc_cov_train,
    #     val_series=val_series,
    #     val_future_covariates=cyc_cov_val,
    #     )

    # cyc_cov_for_forecast = cyc_cov.slice(series_scaled.end_time()- pd.DateOffset(months=config.window_size) + pd.DateOffset(months=1), cyc_cov.end_time())
    fc_scaled = model.predict(n=config.months_to_2099, future_covariates=cyc_cov)

    fc = scaler.inverse_transform(fc_scaled)

    # Clip negative precipitation to 0
    if col == "precip":
        fc = fc.map(lambda x: np.clip(x, 0, None))

    return series, fc

def build_future_covariates(df: pd.DataFrame, config) :

    # Forecast temperature and precipitation
    hist_pr, fc_pr = forecast_covariate_to_2099(df,"precip", config)
    plot_covariate_forecasts(hist_pr, fc_pr, "precip", config, color="green")


    hist_tm, fc_tm = forecast_covariate_to_2099(df, "tm_m", config)
    plot_covariate_forecasts(hist_tm, fc_tm, "tm_m", config, color="blue")
        
        
    # Combine historical covariates
    hist_cov = hist_tm.stack(hist_pr)
    future_cov = fc_tm.stack(fc_pr)

    # build cyclic covariates for entire span (needed to combine with climate covariates)
    start = hist_cov.start_time()
    total_periods = len(hist_pr) + config.months_to_2099
    full_time_idx = pd.date_range(start=start, periods=total_periods, freq="MS")
    time_cov = build_cyclic_covariates(full_time_idx)


    # split time_cov into historical and future parts
    hist_time_cov = time_cov.slice(hist_cov.start_time(), hist_cov.end_time())
    fut_time_cov = time_cov.slice(future_cov.start_time(), future_cov.end_time())


    # stack climate covariates with time covariates for both historical and future
    hist_cov = hist_cov.stack(hist_time_cov)
    future_cov = future_cov.stack(fut_time_cov)


    future_cov = future_cov.slice(hist_cov.end_time() + pd.DateOffset(months=1), future_cov.end_time())
    full_cov = hist_cov.concatenate(future_cov)

    return full_cov


def create_model(model_name: str, config):
    
    if model_name == "ExtraTrees":
        return XGBModel(
            lags=config.window_size,
            output_chunk_length=config.horizon              
            ,lags_past_covariates=[-i for i in range(1,13)],
        )
    elif model_name == "RandomForest":
        return RandomForest(
            n_estimators=100,random_state=config.SEED,
            lags=config.window_size, 
            output_chunk_length=config.horizon                      
            ,lags_past_covariates=[-i for i in range(1,13)]
        )
    elif model_name == "SVR":
        return RegressionModel(
            model=SVR(kernel="rbf", C=1, gamma=0.01, epsilon=0.01),
            lags=config.window_size, 
            output_chunk_length=config.horizon           
            ,lags_past_covariates=[-i for i in range(1,13)]
        )
    elif model_name in ["LSTM","WTLSTM"] :
        return BlockRNNModel(
            model="LSTM", 
            input_chunk_length=config.window_size, 
            output_chunk_length=config.horizon,
            n_epochs=config.num_epochs, 
            optimizer_kwargs={"lr": 1e-3},
            # training_length=48,
            force_reset=True,
            batch_size=16,
            dropout=config.lstm_dropout,
            n_rnn_layers=config.lstm_layers,
            hidden_dim=config.lstm_hidden_dim, 
            random_state=config.SEED,
            # ,likelihood=GaussianLikelihood()
        )
    elif model_name == "NHiTS":
        return NHiTSModel(
            input_chunk_length=config.window_size,
            output_chunk_length=config.horizon,
            n_epochs=config.num_epochs,
            random_state=config.SEED,
        )

    elif model_name == "TFT":
        return TFTModel(
            input_chunk_length=config.window_size,
            output_chunk_length=config.horizon,
            hidden_size=config.lstm_hidden_dim,
            lstm_layers=config.lstm_layers,
            dropout=config.lstm_dropout,
            # attention_dropout=config.lstm_dropout,
            n_epochs=config.num_epochs,
            random_state=config.SEED,
        )
    


def train_and_forecast_spi(hist, full_cov, config, model_name):

    split_point = int(len(hist) * config.train_test_split)
    train_end = hist.time_index[split_point - 1]
    train, test = hist.split_before(train_end + pd.Timedelta(days=1))

    if model_name == "WTLSTM":
        train, full_cov, wavelet_info = prepare_wavelet_data(train, full_cov, return_info=True)
    else:
        wavelet_info = None


    # scaling
    use_scaler = model_name in ["SVR", "LSTM","WTLSTM"]
    if use_scaler:
        scaler = Scaler()
        hist_s = scaler.fit_transform(hist)
        train_s = scaler.fit_transform(train)
    
    
        cov_scaler = Scaler()
        full_cov_s = cov_scaler.fit_transform(full_cov)
    else:
        scaler = None
        hist_s = hist, 
        train_s = train, 
        full_cov_s = full_cov


    model = create_model(model_name, config)

    if model_name == "TFT":
         model.fit(train_s, future_covariates=full_cov_s)
    else:
        model.fit(train_s, past_covariates=full_cov_s)

    
    if model_name == "TFT":
        pred_scaled = model.predict(n=len(test), future_covariates=full_cov_s)

    else:
        pred_scaled = model.predict(n=len(test), past_covariates=full_cov_s)


    pred = scaler.inverse_transform(pred_scaled) if use_scaler else pred_scaled


    metrics = calculate_metrics(test.values(), pred.values())
    
    # model = create_model(model_name, config)

    #refit
    if model_name == "TFT":
        model.fit(hist_s, future_covariates=full_cov_s)
    else:
        model.fit(hist_s, past_covariates=full_cov_s)

    if model_name == "TFT":
        fc_scaled = model.predict(n=config.months_to_2099, future_covariates=full_cov_s)
    else:
        fc_scaled = model.predict(n=config.months_to_2099, past_covariates=full_cov_s)

    fc = scaler.inverse_transform(fc_scaled) if use_scaler else fc_scaled



    return {
    "metrics": metrics,
    "test_pred": pred,
    "test_obs": test,
    "forecast_to_2099": fc,
    }

class ForecastConfig:
    def __init__(self):
        self.SEED = 42
        self.horizon =  3
        self.window_size = 15
        self.num_epochs = 170
        self.input_folder = "./Data/maindata"
        self.SPI = ["SPI_1", "SPI_3", "SPI_6", "SPI_9", "SPI_12", "SPI_24"]
        self.models_to_test = ["ExtraTrees", "RandomForest", "SVR", "LSTM","WTLSTM","NHiTS", "TFT"]
        self.train_test_split = 0.8
        self.lstm_hidden_dim = 64
        self.lstm_dropout = 0.1
        self.lstm_layers = 2

        ts = datetime.datetime.now().strftime("%m%d%H%M")
        self.output_folder = (
            f"./Results/{ts}_e{self.num_epochs}"
            f"-hd{self.lstm_hidden_dim}-l{self.lstm_layers}"
            f"-d{self.lstm_dropout}-h{self.horizon}"
        )

        os.makedirs(self.output_folder, exist_ok=True)
        np.random.seed(self.SEED)


def plot_each_forecast(config, station, spi, model_name, hist, pred, fc, metrics_text):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(hist.time_index,hist.values(),label="Historical",lw=0.8)
    ax.plot(pred.time_index,pred.values(),label="Predicted (Test)",lw=0.8,linestyle="--",color="red")
    ax.set_title(f"{station} {spi} {model_name} ")
    ax.set_xlabel("Date")
    ax.set_ylabel(spi)
    ax.text(0.02, 0.95, metrics_text,transform=ax.transAxes,fontsize=10,verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",facecolor="white", alpha=0.7))
    ax.axvspan(pred.time_index.min(), pred.time_index.max(),color='gray', alpha=0.1, label="Test period")
                
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    outfile = os.path.join(config.output_folder,f"{station}_{spi}_{model_name}_historical.png")
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()                    

                # Plot results
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(hist.time_index, hist.values(), label="Historical", lw=0.6)
    ax.plot(pred.time_index, pred.values(), label="Predicted", lw=0.4,color="red", linestyle="--")
    ax.plot(fc.time_index, fc.values(), label="Forecast",lw=0.6, color="green")
    ax.set_title(f"{station} {spi} {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel(spi)
    ax.grid(True)
    ax.legend()

    ax.text(
                    0.02, 0.95, metrics_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                            facecolor="white", alpha=0.7)
                )

                # ----------------------- Global trend line -----------------------
    forecast_df = fc.to_dataframe()
    x = np.arange(len(forecast_df))
    y = forecast_df.iloc[:, 0].values
    dates = forecast_df.index

    coef = np.polyfit(x, y, 1)
    global_trend = np.polyval(coef, x)
    m_global, b_global = coef

    ax.plot(
                    dates,
                    global_trend,
                    label="Global Trend",
                    linestyle="--",
                    linewidth=2,
                    color="blue",
                )

                # --- Place trend equation bottom-right near the line ---
    y_last = global_trend[-1]

                # Offsets based on visible range
    x_offset = (dates[-1] - dates[0]) * 0.03
    y_offset = (max(global_trend) - min(global_trend)) * -0.05

    equation = f"y = {m_global:.4f}x + {b_global:.4f}"

    ax.annotate(
                    equation,
                    xy=(dates[-1], y_last),
                    xytext=(dates[-1] - x_offset, y_last + y_offset),
                    fontsize=10,
                    color="blue",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

                # ----------------------- Decade trend lines -----------------------
    decade_length = 240  # 10 years
    n = len(y)

    for start in range(0, n, decade_length):
        end = min(start + decade_length, n)
        x_dec = x[start:end]
        y_dec = y[start:end]
        date_dec = dates[start:end]

        coef_dec = np.polyfit(x_dec, y_dec, 1)
        trend_dec = np.polyval(coef_dec, x_dec)

        ax.plot(
                        date_dec,
                        trend_dec,
                        linestyle="-",
                        linewidth=2,
                        alpha=0.9,
                        # label=f"Decade Trend {date_dec[0].year}-{date_dec[-1].year}"
                    )

    ax.legend()

                # ----------------------- Save -----------------------
    outfile = os.path.join(config.output_folder,
                                    f"{station}_{spi}_{model_name}.png")
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    config = ForecastConfig()
    all_results = []
    
    data_files = glob.glob(os.path.join(config.input_folder, "*.csv"))
    
    
    for file in data_files:
        station = os.path.splitext(os.path.basename(file))[0]
        # if station != "40700":
        #     continue
        print(f"Processing station: {station}")
        
        df = pd.read_csv(file, parse_dates=["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        df = df.set_index("ds").asfreq("MS").reset_index()
        # plot_raw_data(df, station, config)

        last_date = df['ds'].max()
        config.months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month+1)
        config.station = station

        full_cov = build_future_covariates(df, config)

        best_results = []
        for spi in config.SPI:
            # if spi not in ["SPI_6"]:
            #     continue
            model_metrics = []
            for model_name in config.models_to_test:
                # if model_name != "LSTM":
                #     continue
                print(f"Running: {station} {spi} {model_name}")

                df_spi = df[["ds", spi]].dropna().sort_values("ds").reset_index(drop=True)
                hist = TimeSeries.from_dataframe(df_spi, time_col="ds", value_cols=spi)
                results = train_and_forecast_spi(hist, full_cov, config, model_name)     
                test = results["test_obs"]
                pred = results["test_pred"]
                fc = results["forecast_to_2099"]
                metrics = results["metrics"]

                # plot_scatter(test, pred, station, spi, model_name, config)
                # plot_residual_distribution(test, pred, station, spi, model_name, config)
                # plot_rolling_error(test, pred, test.time_index, station, spi, model_name, config)
                
                

                corr_val = metrics['corr'][0]
                rmse_val = metrics['rmse']

                # metrics_text = f"RMSE: {rmse_val:.3f}\nCorr: {corr_val:.3f}"
                
                # plot_each_forecast(config, station, spi, model_name, hist, pred, fc, metrics_text)

                res = {
                    "spi": spi,
                    "model": model_name,
                    **metrics,
                    "horizon": config.horizon,
                    "window_size": config.window_size,
                    "epoch": config.num_epochs,
                    "forecast": fc,
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
        if best_results:
            plot_final_forecasts(station, best_results, os.path.join(config.output_folder, f"forecast_{station}.png"))
            plot_heatmaps(station, best_results, os.path.join(config.output_folder, f"heatmap_{station}.png"))
            
       
    
    # Save metrics and create Taylor diagrams
    if all_results:
        metrics_df = pd.DataFrame(all_results)
        metrics_df.to_csv(os.path.join(config.output_folder, "summary_metrics.csv"), index=False)
        
        for station in metrics_df["station"].unique():
            taylor_diagram_panel(config,metrics_df, station, os.path.join(config.output_folder, f"taylor_{station}.png"))
        # plot_metric_boxplots(metrics_df, config)
        # plot_model_ranking(metrics_df, config)



                


               
    
    print(f"✅ Done! Results saved in: {config.output_folder}")


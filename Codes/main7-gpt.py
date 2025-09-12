# -----------------------------
# Imports and Configuration
# -----------------------------
import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import seaborn as sns

# Machine learning imports
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Time series analysis imports
from darts import TimeSeries
from darts.models import BlockRNNModel, RegressionModel
# from darts.metrics import mae, mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.statistics import check_seasonality , plot_acf

# Wavelet transform
import pywt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration Class
# -----------------------------
class ForecastConfig:
    """Configuration class for forecasting parameters"""
    def __init__(self):
        self.SEED = 42
        self.horizon = 3
        self.num_epochs = 100
        self.input_folder = "./Data/maindata"
        self.output_folder = "./Results/r37"
        # self.SPI = ["SPI_1", "SPI_3", "SPI_6", "SPI_9", "SPI_12", "SPI_24"]
        self.SPI = ["SPI_1"]
        # self.models_to_test = ["ExtraTrees", "WTLSTM", "RandomForest", "SVR", "LSTM"]
        self.models_to_test = [ "ExtraTrees"]
        self.wavelet = "db4"
        self.wavelet_level = 1
        self.train_test_split = 0.8
        self.lstm_hidden_dim = 64
        self.lstm_batch_size = 16
        self.lstm_dropout = 0.1
        self.lstm_layers = 1
        
        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(self.SEED)

# -----------------------------
# Utility Functions
# -----------------------------
def setup_plotting():
    """Configure matplotlib settings"""
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True


def wavelet_denoise(series: np.ndarray, wavelet: str = "db4", level: int = 1) -> np.ndarray:
    """
    Denoise a 1D numpy array using wavelet thresholding
    
    Args:
        series: 1D array to denoise
        wavelet: Wavelet type to use
        level: Decomposition level
        
    Returns:
        Denoised array of the same length
    """
    coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
    # Universal threshold based on noise estimate
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(series)))
    coeffs_denoised = [pywt.threshold(c, threshold, mode="soft") if i > 0 else c
                       for i, c in enumerate(coeffs)]
    denoised = pywt.waverec(coeffs_denoised, wavelet=wavelet)
    return denoised[:len(series)]

def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for forecasts
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
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
        logger.warning("Cannot calculate CRMSE due to zero standard deviation")
    
    return {
        "std_ref": std_ref,
        "std_model": std_sim,
        "rmse": rmse_val,
        "corr": corr_val,
        "crmse": crmse_val,
        "mae": mae_val,
        "mape": mape_val
    }

# -----------------------------
# Core Forecasting Functions
# -----------------------------
def forecast_covariate_to_2099(df: pd.DataFrame, col: str, last_date: datetime, config: ForecastConfig) -> Tuple[TimeSeries, TimeSeries]:
    """
    Forecast a covariate (temperature or precipitation) to the year 2099
    
    Args:
        df: DataFrame containing historical data
        col: Column name to forecast
        last_date: Last date in the historical data
        config: Configuration object
        
    Returns:
        Tuple of (historical series, forecast series)
    """
    try:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.set_index("ds").asfreq("MS").reset_index()

        scaler = Scaler()
        series = TimeSeries.from_dataframe(df, 'ds', col)

        # Check seasonality
        is_seasonal, period = check_seasonality(series, max_lag=120)
        logger.info(f"Seasonality check: {is_seasonal}, period: {period}")
        
        # Plot ACF
        plot_acf(series, period, max_lag=120)
        outfile = os.path.join(config.output_folder, f"acf_plot_{col}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        window_size = int(period) if period > 0 else 12  # Default to 12 months if no seasonality detected

        series_scaled = scaler.fit_transform(series)

        # Build cyclic covariates
        month_cov = datetime_attribute_timeseries(series.time_index, "month", one_hot=True)
        year_cov = datetime_attribute_timeseries(series.time_index, "year")
        cyc_cov = month_cov.stack(year_cov)
        cov_scaler = Scaler()
        cyc_cov = cov_scaler.fit_transform(cyc_cov)

        # Initialize and train model
        model = BlockRNNModel(
            model='LSTM',
            input_chunk_length=window_size,
            output_chunk_length=config.horizon,
            n_epochs=config.num_epochs,
            dropout=config.lstm_dropout,
            hidden_dim=config.lstm_hidden_dim,
            batch_size=config.lstm_batch_size,
            random_state=config.SEED
        )

        model.fit(series_scaled, past_covariates=cyc_cov)

        # Calculate months to forecast until 2099
        months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month)
        if months_to_2099 <= 0:
            logger.warning("No future forecasting needed - already beyond 2099")
            return series, TimeSeries.from_times_and_values(
                pd.date_range(start=last_date, periods=1, freq="MS"), 
                [series.values()[-1][0]]
            )

        # Create future covariates for prediction
        future_time_idx = pd.date_range(
            start=series.time_index[0], 
            periods=len(series) + months_to_2099, 
            freq="MS"
        )
        month_future = datetime_attribute_timeseries(future_time_idx, "month", one_hot=True)
        year_future = datetime_attribute_timeseries(future_time_idx, "year")
        cyc_cov_full = month_future.stack(year_future)
        cyc_cov_full = cov_scaler.transform(cyc_cov_full)

        # Predict future
        fc_scaled = model.predict(
            n=months_to_2099, 
            series=series_scaled, 
            past_covariates=cyc_cov_full
        )
        fc = scaler.inverse_transform(fc_scaled)

        return series, fc

    except Exception as e:
        logger.error(f"Error forecasting {col}: {e}")
        raise

def build_future_covariates(df: pd.DataFrame, last_date: datetime, config: ForecastConfig) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    """
    Build future covariates for forecasting
    
    Args:
        df: DataFrame containing historical data
        last_date: Last date in the historical data
        config: Configuration object
        
    Returns:
        Tuple of (full future covariates, historical covariates, future covariates)
    """
    try:
        # Forecast temperature and precipitation
        hist_tm, fc_tm = forecast_covariate_to_2099(df, "tm_m", last_date, config)
        hist_pr, fc_pr = forecast_covariate_to_2099(df, "precip", last_date, config)

        # Combine historical covariates
        hist_ts = hist_tm.stack(hist_pr)

        # Create future covariates DataFrame
        future_df = pd.DataFrame({
            "ds": fc_tm.time_index,
            "tm_m": fc_tm.values().flatten(),
            "precip": fc_pr.values().flatten()
        })
        future_ts = TimeSeries.from_dataframe(future_df, "ds", ["tm_m", "precip"])

        # Combine historical and future covariates
        full_future_cov = hist_ts.concatenate(future_ts)

        return full_future_cov, hist_ts, future_ts

    except Exception as e:
        logger.error(f"Error building future covariates: {e}")
        raise

def plot_covariate_forecasts(hist_ts: TimeSeries, future_ts: TimeSeries, covariate: str, 
                            station: str, config: ForecastConfig, color: str = "blue"):
    """
    Plot historical vs forecasted covariate
    
    Args:
        hist_ts: Historical time series
        future_ts: Forecasted time series
        covariate: Name of the covariate
        station: Station identifier
        config: Configuration object
        color: Plot color
    """
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(hist_ts.time_index, hist_ts[covariate].values().flatten(), 
                label=f"Historical {covariate}", color=color, lw=0.5)
        plt.plot(future_ts.time_index, future_ts[covariate].values().flatten(), 
                label=f"Forecast {covariate}", color=color, lw=0.5)
        plt.axvline(x=hist_ts.end_time(), color="red", linestyle=":", lw=1.5, label="Forecast Start")
        plt.title(f"{station} - {covariate} - Historical vs Forecasted to 2099")
        plt.xlabel("Date")
        plt.ylabel(covariate)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        outfile = os.path.join(config.output_folder, f"covariate_{covariate}_{station}.png") 
        plt.savefig(outfile, dpi=300, bbox_inches="tight") 
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting covariate forecasts: {e}")

def prepare_wavelet_data(train_series: TimeSeries, test_series: TimeSeries, 
                        train_cov: TimeSeries, test_cov: TimeSeries, 
                        value_col: str) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
    """
    Prepare wavelet-denoised data for WTLSTM model
    
    Args:
        train_series: Training time series
        test_series: Testing time series
        train_cov: Training covariates
        test_cov: Testing covariates
        value_col: Value column name
        
    Returns:
        Tuple of denoised training series, testing series, training covariates, and testing covariates
    """
    # Convert to DataFrames
    train_df = train_series.to_dataframe().reset_index()
    test_df = test_series.to_dataframe().reset_index()
    traincov_df = train_cov.to_dataframe().reset_index()
    testcov_df = test_cov.to_dataframe().reset_index()

    # Apply wavelet denoising
    train_df[f"{value_col}_denoised"] = wavelet_denoise(train_df[value_col].values)
    test_df[f"{value_col}_denoised"] = wavelet_denoise(test_df[value_col].values)
    traincov_df["tm_m_denoised"] = wavelet_denoise(traincov_df["tm_m"].values)
    testcov_df["tm_m_denoised"] = wavelet_denoise(testcov_df["tm_m"].values)
    traincov_df["precip_denoised"] = wavelet_denoise(traincov_df["precip"].values)
    testcov_df["precip_denoised"] = wavelet_denoise(testcov_df["precip"].values)

    # Get cyclic columns
    cyclic_cols = [col for col in traincov_df.columns if "month" in col or "year" in col]

    # Rebuild TimeSeries objects
    train_denoised = TimeSeries.from_dataframe(train_df, "ds", f"{value_col}_denoised")
    test_denoised = TimeSeries.from_dataframe(test_df, "ds", f"{value_col}_denoised")
    train_cov_denoised = TimeSeries.from_dataframe(
        traincov_df, "ds", ["tm_m_denoised", "precip_denoised"] + cyclic_cols
    )
    test_cov_denoised = TimeSeries.from_dataframe(
        testcov_df, "ds", ["tm_m_denoised", "precip_denoised"] + cyclic_cols
    )

    return train_denoised, test_denoised, train_cov_denoised, test_cov_denoised

def create_model(model_name: str, window_size: int, config: ForecastConfig):
    """
    Create a forecasting model based on the specified type
    
    Args:
        model_name: Type of model to create
        window_size: Input window size
        config: Configuration object
        
    Returns:
        Initialized model
    """
    if model_name == "ExtraTrees":
        return RegressionModel(
            ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=config.SEED, n_jobs=-1),
            lags=window_size, 
            output_chunk_length=config.horizon
        )
    elif model_name == "RandomForest":
        return RegressionModel(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=config.SEED, n_jobs=-1),
            lags=window_size, 
            output_chunk_length=config.horizon
        )
    elif model_name == "SVR":
        return RegressionModel(
            SVR(kernel="rbf", C=1, gamma=0.01, epsilon=0.01),
            lags=window_size, 
            output_chunk_length=config.horizon
        )
    elif model_name == "LSTM":
        return BlockRNNModel(
            model="LSTM", 
            input_chunk_length=window_size, 
            output_chunk_length=config.horizon,
            n_epochs=config.num_epochs, 
            dropout=config.lstm_dropout,
            n_rnn_layers=config.lstm_layers,
            hidden_dim=config.lstm_hidden_dim, 
            batch_size=config.lstm_batch_size, 
            random_state=config.SEED,
            likelihood=GaussianLikelihood()
        )
    elif model_name == "WTLSTM":
        return BlockRNNModel(
            model='LSTM', 
            input_chunk_length=window_size, 
            output_chunk_length=config.horizon,
            n_epochs=config.num_epochs, 
            dropout=config.lstm_dropout,
            n_rnn_layers=config.lstm_layers,
            hidden_dim=config.lstm_hidden_dim, 
            batch_size=config.lstm_batch_size, 
            random_state=config.SEED,
            likelihood=GaussianLikelihood()
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def train_and_forecast(df: pd.DataFrame, value_col: str,station:str, model_name: str, 
                      full_future_cov: TimeSeries, config: ForecastConfig) -> Dict[str, Any]:
    """
    Train a model and forecast future values
    
    Args:
        df: DataFrame containing the data
        value_col: Column to forecast
        model_name: Type of model to use
        full_future_cov: Future covariates
        config: Configuration object
        
    Returns:
        Dictionary with results and metrics
    """
    try:
        # Prepare data
        df_spi = df[["ds", value_col, "tm_m", "precip"]].dropna().reset_index(drop=True)
        
        if len(df_spi) == 0:
            logger.warning(f"No data available for {value_col}")
            return None

        hist_series = TimeSeries.from_dataframe(df_spi, 'ds', value_col)
        hist_covariates = TimeSeries.from_dataframe(df_spi, 'ds', ["tm_m", "precip"])

        # Build cyclic covariates
        month_cov = datetime_attribute_timeseries(hist_series.time_index, "month", one_hot=True)
        year_cov = datetime_attribute_timeseries(hist_series.time_index, "year")
        cyc_cov = month_cov.stack(year_cov)
        hist_covariates = hist_covariates.stack(cyc_cov)

        # Check seasonality
        is_seasonal, period = check_seasonality(hist_series, max_lag=120)
        logger.info(f"Seasonality check for {value_col}: {is_seasonal}, period: {period}")
        
        # Plot ACF
        plot_acf(hist_series, period, max_lag=120)
        outfile = os.path.join(config.output_folder, f"acf_plot_{station} {value_col}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        window_size = int(period) if period > 36 else 12

        # Split data
        train, test = hist_series.split_before(config.train_test_split)
        train_cov, test_cov = hist_covariates.split_before(config.train_test_split)
        test_raw = test.copy()

        # Handle wavelet denoising for WTLSTM model
        if model_name == "WTLSTM":
            train, test, train_cov, test_cov = prepare_wavelet_data(
                train, test, train_cov, test_cov, value_col
            )

        # Scale data if needed
        use_scaler = model_name in ["SVR", "LSTM", "WTLSTM"]
        scaler = None
        cov_scaler = None
        
        if use_scaler:
            scaler = Scaler()
            hist_series_scaled = scaler.fit_transform(hist_series)
            train_scaled = scaler.fit_transform(train)
            
            cov_scaler = Scaler()
            train_cov_scaled = cov_scaler.fit_transform(train_cov)
            hist_covariates_scaled = cov_scaler.fit_transform(hist_covariates)
        else:
            hist_series_scaled = hist_series
            train_scaled = train
            train_cov_scaled = train_cov
            hist_covariates_scaled = hist_covariates

        # Create and train model
        model = create_model(model_name, window_size, config)
        
        if model_name in ["SVR", "RandomForest", "ExtraTrees"]:
            model.fit(train_scaled)
        else:
            model.fit(train_scaled, past_covariates=train_cov_scaled)

        # Make predictions
        if model_name in ["SVR", "RandomForest", "ExtraTrees"]:
            pred = model.predict(n=len(test), series=train_scaled)
        else:
            pred = model.predict(n=len(test), series=train_scaled, past_covariates=hist_covariates_scaled)

        # Inverse transform if scaled
        if use_scaler:
            pred = scaler.inverse_transform(pred)
            test = scaler.inverse_transform(test)

        # Calculate metrics
        observed = test_raw.values().flatten()
        predicted = pred.values().flatten()
        metrics = calculate_metrics(observed, predicted)

        # Refit model on full historical data
        if model_name in ["SVR", "RandomForest", "ExtraTrees"]:
            model.fit(hist_series_scaled)
        else:
            model.fit(hist_series_scaled, past_covariates=hist_covariates_scaled)

        # Forecast to 2099
        last_date = df["ds"].max()
        months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month)
        
        if months_to_2099 <= 0:
            logger.warning("No future forecasting needed - already beyond 2099")
            forecast = TimeSeries.from_times_and_values(
                pd.date_range(start=last_date, periods=1, freq="MS"), 
                [hist_series.values()[-1][0]]
            )
        else:
            # Prepare future covariates
            future_time_idx = pd.date_range(
                start=full_future_cov.time_index[0], 
                periods=len(full_future_cov), 
                freq="MS"
            )
            month_future = datetime_attribute_timeseries(future_time_idx, "month", one_hot=True)
            year_future = datetime_attribute_timeseries(future_time_idx, "year")
            cyc_cov_full = month_future.stack(year_future)
            
            # Combine with existing covariates
            full_future_cov_combined = full_future_cov.stack(cyc_cov_full)
            
            if cov_scaler is not None:
                full_future_cov_combined = cov_scaler.transform(full_future_cov_combined)

            # Make forecast
            if model_name in ["SVR", "RandomForest", "ExtraTrees"]:
                forecast = model.predict(n=months_to_2099, series=hist_series_scaled)
            else:
                forecast = model.predict(
                    n=months_to_2099,
                    series=hist_series_scaled,
                    past_covariates=full_future_cov_combined,
                    num_samples=100
                )

            if scaler is not None:
                forecast = scaler.inverse_transform(forecast)
                hist_series_scaled = scaler.inverse_transform(hist_series_scaled)

        # Plot results
        plt.figure(figsize=(16, 6))
        plt.plot(df['ds'], df[value_col], label="Historical", lw=0.6)
        plt.plot(pred.time_index, predicted, label="Predicted", lw=0.4, color="red", linestyle="--")
        plt.plot(forecast.time_index, forecast.values(), label="Forecast", lw=0.6, color="green")
        plt.title(f"{station} {value_col} {model_name} Forecast till 2099")
        plt.xlabel("Date")
        plt.ylabel(value_col)
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
        
        outfile = os.path.join(config.output_folder, f"{station} {value_col}_{model_name}.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "spi": value_col,
            "model": model_name,
            **metrics,
            "horizon": config.horizon,
            "window_size": window_size,
            "epoch": config.num_epochs,
            "forecast": forecast,
            "pred": pred,
            "series": hist_series
        }

    except Exception as e:
        logger.error(f"Error in train_and_forecast for {model_name}: {e}")
        return None

def pick_best_model(models: List[Dict], weights: Dict[str, float] = None) -> Dict:
    """
    Pick best model using multiple metrics
    
    Args:
        models: List of model results dictionaries
        weights: Dictionary of metric weights
        
    Returns:
        Best model dictionary
    """
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
        # scaler = res["scaler"]
        forecast = res["forecast"]
        # Historical
        axes[i].plot(df["ds"], df[df.columns[1]], lw=0.6, alpha=0.7, label="Historical")

        # Prediction
        # if res["pred"] is not None:
            # if scaler:
        # pred_inv = scaler.transform(res["pred"])
        p = res["pred"].values().flatten()

        axes[i].plot(res["pred"].time_index, p, lw=0.7, color="red", label="Prediction")

        # Forecast
        # if scaler:
        #     forecast_inv = scaler.inverse_transform(res["forecast"])
        #     f = forecast_inv.values().flatten()
        # else:
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
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# Main Execution
# -----------------------------
def main():
    """Main execution function"""
    config = ForecastConfig()
    setup_plotting()
    all_results = []
    
    # Get list of data files
    data_files = glob.glob(os.path.join(config.input_folder, "*.csv"))
    if not data_files:
        logger.warning(f"No CSV files found in {config.input_folder}")
        return
    
    for file in data_files:
        station = os.path.splitext(os.path.basename(file))[0]
        if station !="40700":
            continue
        logger.info(f"Processing station: {station}")
        
        try:
            # Load data
            df = pd.read_csv(file, parse_dates=["ds"])
            if df.empty:
                logger.warning(f"Empty DataFrame for station {station}")
                continue
                
            last_date = df['ds'].max()
            
            # Build future covariates
            future_covariates_ts, hist_ts, future_ts = build_future_covariates(df, last_date, config)
            plot_covariate_forecasts(hist_ts, future_ts, "tm_m", station, config, color="blue")
            plot_covariate_forecasts(hist_ts, future_ts, "precip", station, config, color="green")
            full_cov = hist_ts.concatenate(future_ts)

            best_results = []
            for spi in config.SPI:
                model_metrics = []
                for model_name in config.models_to_test:
                    logger.info(f"Running: {station} {spi} {model_name}")
                    
                    res = train_and_forecast(df, spi, station,model_name, full_cov, config)
                    if res:
                        model_metrics.append(res)
                        all_results.append({
                            "station": station,
                            **{k: v for k, v in res.items() if k not in ["forecast", "pred", "series", "scaler"]}
                        })
                
                # Select best model for this SPI
                best_model = pick_best_model(model_metrics)
                if best_model:
                    best_results.append(best_model)
            
            # Save plots for this station
            if best_results:
                plot_final_forecasts(station, best_results, os.path.join(config.output_folder, f"forecast_{station}.png"))
                plot_heatmaps(station, best_results, os.path.join(config.output_folder, f"heatmap_{station}.png"))
                
        except Exception as e:
            logger.error(f"Error processing station {station}: {e}")
    
    # Save metrics and create Taylor diagrams
    if all_results:
        metrics_df = pd.DataFrame(all_results)
        metrics_df.to_csv(os.path.join(config.output_folder, "summary_metrics.csv"), index=False)
        
        for station in metrics_df["station"].unique():
            taylor_diagram_panel(config,metrics_df, station, os.path.join(config.output_folder, f"taylor_{station}.png"))
    
    logger.info(f"✅ Done! Results saved in: {config.output_folder}")

if __name__ == "__main__":
    main()
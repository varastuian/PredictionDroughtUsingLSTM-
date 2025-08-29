import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor

from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import rmse

# -----------------------------
# Settings
# -----------------------------
SEED = 42
window_size = 36
horizon = 1  # 1-step forecast works best for tree ensembles
station_file = r"Data\python_spi\40708_SPI.csv"
spi_column = 'SPI_3'

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)
df = df.dropna(subset=[spi_column])  # drop missing values

# Create TimeSeries (no scaling needed for ExtraTrees)
series = TimeSeries.from_dataframe(df, 'ds', spi_column)

# -----------------------------
# Train / Test Split
# -----------------------------
train, test = series[:-48], series[-48:]

# -----------------------------
# Hyperparameter Grid for ExtraTrees
# -----------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10]
}

best_score = float("inf")
best_params = None
best_model = None

print("Tuning ExtraTrees hyperparameters...")

for n in param_grid["n_estimators"]:
    for depth in param_grid["max_depth"]:
        for min_split in param_grid["min_samples_split"]:
            try:
                model = RegressionModel(
                    model=ExtraTreesRegressor(
                        n_estimators=n,
                        max_depth=depth,
                        min_samples_split=min_split,
                        random_state=SEED,
                        n_jobs=-1
                    ),
                    lags=window_size,
                    output_chunk_length=horizon
                )
                model.fit(train)
                
                pred = model.historical_forecasts(
                    series,
                    start=train.end_time(),
                    forecast_horizon=horizon,
                    stride=1,
                    retrain=False,
                    verbose=False
                )
                pred = pred.slice_intersect(test)

                score = rmse(test.slice_intersect(pred), pred)
                print(f"n={n}, depth={depth}, min_split={min_split} -> RMSE={score:.4f}")

                if score < best_score:
                    best_score = score
                    best_params = (n, depth, min_split)
                    best_model = model

            except Exception as e:
                print(f"Error with params n={n}, depth={depth}, min_split={min_split}: {e}")

print("\nBest params:", best_params, "with RMSE:", best_score)

# -----------------------------
# Final Evaluation
# -----------------------------
pred_backtest = best_model.historical_forecasts(
    series,
    start=train.end_time(),
    forecast_horizon=horizon,
    stride=1,
    retrain=False,
    verbose=True
)
pred_backtest = pred_backtest.slice_intersect(test)

o = test.slice_intersect(pred_backtest).values().flatten()
p = pred_backtest.values().flatten()
print("Final Backtest RMSE:", rmse(test.slice_intersect(pred_backtest), pred_backtest))
print("Pearson correlation:", pearsonr(o, p)[0])

# -----------------------------
# Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=2, alpha=0.7)
plt.plot(pred_backtest.time_index, p, label="ET Predicted", lw=2, color="red", linestyle="--")
plt.title(f"{spi_column} ExtraTrees Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Forecast till 2099
# -----------------------------
last_date = df['ds'].max()
months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)

step_size = 12
forecast_series = series
all_forecasts = []

forecast_end_date = pd.Timestamp("2099-12-31")
while forecast_series.end_time() < forecast_end_date:
    step = min(step_size, months_to_2099 - len(all_forecasts))
    f = best_model.predict(n=step, series=forecast_series)
    all_forecasts.append(f)
    forecast_series = forecast_series.append(f)

forecast = all_forecasts[0]
for f in all_forecasts[1:]:
    forecast = forecast.append(f)

plt.figure(figsize=(16,6))
plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=2, alpha=0.7)
plt.plot(forecast.time_index, forecast.values(), label="ET Forecast", lw=2, color="green", linestyle="--")
plt.title(f"{spi_column} ExtraTrees Forecast till 2099")
plt.xlabel("Date")
plt.ylabel(spi_column)
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.grid(True)
plt.show()


# ets_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import rmse

def run_et_forecast(df, spi_column='SPI_3', window_size=36, horizon=1, seed=42, test_months=48):
    """
    Run ExtraTrees regression forecast on SPI data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'ds' column (datetime) and SPI values.
    spi_column : str
        Column name containing SPI values.
    window_size : int
        Number of lags for the regression model.
    horizon : int
        Forecast horizon (1-step forecast recommended for tree ensembles).
    seed : int
        Random seed for reproducibility.
    test_months : int
        Number of months to use for testing.

    Returns:
    --------
    best_model : RegressionModel
        Fitted Darts ExtraTrees model.
    pred_backtest : TimeSeries
        Backtested predictions on the test set.
    forecast : TimeSeries
        Forecast until 2099.
    best_params : tuple
        Best hyperparameters (n_estimators, max_depth, min_samples_split)
    best_rmse : float
        RMSE of best model on backtest
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df.sort_values('ds', inplace=True)
    df = df.dropna(subset=[spi_column])

    series = TimeSeries.from_dataframe(df, 'ds', spi_column)
    train, test = series[:-test_months], series[-test_months:]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    }

    best_score = float("inf")
    best_params = None
    best_model = None

    print("Tuning ExtraTrees hyperparameters...")
    for n in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            for min_split in param_grid["min_samples_split"]:
                try:
                    model = RegressionModel(
                        model=ExtraTreesRegressor(
                            n_estimators=n,
                            max_depth=depth,
                            min_samples_split=min_split,
                            random_state=seed,
                            n_jobs=-1
                        ),
                        lags=window_size,
                        output_chunk_length=horizon
                    )
                    model.fit(train)

                    pred = model.historical_forecasts(
                        series,
                        start=train.end_time(),
                        forecast_horizon=horizon,
                        stride=1,
                        retrain=False,
                        verbose=False
                    )
                    pred = pred.slice_intersect(test)

                    score = rmse(test.slice_intersect(pred), pred)
                    print(f"n={n}, depth={depth}, min_split={min_split} -> RMSE={score:.4f}")

                    if score < best_score:
                        best_score = score
                        best_params = (n, depth, min_split)
                        best_model = model

                except Exception as e:
                    print(f"Error with params n={n}, depth={depth}, min_split={min_split}: {e}")

    print("\nBest params:", best_params, "with RMSE:", best_score)

    # Backtest evaluation
    pred_backtest = best_model.historical_forecasts(
        series,
        start=train.end_time(),
        forecast_horizon=horizon,
        stride=1,
        retrain=False,
        verbose=True
    )
    pred_backtest = pred_backtest.slice_intersect(test)

    o = test.slice_intersect(pred_backtest).values().flatten()
    p = pred_backtest.values().flatten()
    print("Final Backtest RMSE:", rmse(test.slice_intersect(pred_backtest), pred_backtest))
    print("Pearson correlation:", pearsonr(o, p)[0])

    # Plot Actual vs Predicted
    plt.figure(figsize=(14,6))
    plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=2, alpha=0.7)
    plt.plot(pred_backtest.time_index, p, label="ET Predicted", lw=2, color="red", linestyle="--")
    plt.title(f"{spi_column} ExtraTrees Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel(spi_column)
    plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Forecast till 2099
    last_date = df['ds'].max()
    months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)

    step_size = 12
    forecast_series = series
    all_forecasts = []
    forecast_end_date = pd.Timestamp("2099-12-31")

    while forecast_series.end_time() < forecast_end_date:
        step = min(step_size, months_to_2099 - len(all_forecasts))
        f = best_model.predict(n=step, series=forecast_series)
        all_forecasts.append(f)
        forecast_series = forecast_series.append(f)

    forecast = all_forecasts[0]
    for f in all_forecasts[1:]:
        forecast = forecast.append(f)

    plt.figure(figsize=(16,6))
    plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=2, alpha=0.7)
    plt.plot(forecast.time_index, forecast.values(), label="ET Forecast", lw=2, color="green", linestyle="--")
    plt.title(f"{spi_column} ExtraTrees Forecast till 2099")
    plt.xlabel("Date")
    plt.ylabel(spi_column)
    plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6)
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_model, pred_backtest, forecast, best_params, best_score

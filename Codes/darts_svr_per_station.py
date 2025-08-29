import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import rmse

# -----------------------------
# Settings
# -----------------------------
SEED = 42
window_size = 36
horizon = 1  # SVR works best for 1-step forecasts
station_file = r"Data\python_spi\40726_SPI.csv"
spi_column = 'SPI_24'

# -----------------------------
# Load SPI data
# -----------------------------
df = pd.read_csv(station_file)
df["ds"] = pd.to_datetime(df["ds"])
df.sort_values("ds", inplace=True)

# Drop rows where SPI is NaN
df = df.dropna(subset=[spi_column])

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
df['spi_scaled'] = scaler.fit_transform(df[[spi_column]])

# Create TimeSeries
series = TimeSeries.from_dataframe(df, 'ds', 'spi_scaled')

# -----------------------------
# Train / Test Split
# -----------------------------
train, test = series[:-48], series[-48:]

#  -----------------------------
# Hyperparameter Grid for SVR
# -----------------------------
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "epsilon": [0.001, 0.01, 0.1, 0.5]
}

best_score = float("inf")
best_params = None
best_model = None

print("Tuning SVR hyperparameters...")

for C in param_grid["C"]:
    for gamma in param_grid["gamma"]:
        for epsilon in param_grid["epsilon"]:
            try:
                model = RegressionModel(
                    model=SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon),
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
                print(f"C={C}, gamma={gamma}, epsilon={epsilon} -> RMSE={score:.4f}")

                if score < best_score:
                    best_score = score
                    best_params = (C, gamma, epsilon)
                    best_model = model

            except Exception as e:
                print(f"Error with params C={C}, gamma={gamma}, epsilon={epsilon}: {e}")

print("\nBest params:", best_params, "with RMSE:", best_score)

# -----------------------------
# Final Evaluation (historical_forecasts)
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

# Inverse transform for evaluation
o = scaler.inverse_transform(test.slice_intersect(pred_backtest).values().reshape(-1,1)).flatten()
p = scaler.inverse_transform(pred_backtest.values().reshape(-1,1)).flatten()

print("Final Backtest RMSE:", rmse(test.slice_intersect(pred_backtest), pred_backtest))
print("Pearson correlation:", pearsonr(o, p)[0])


# ----------------------------- # Plot Actual vs Predicted # ----------------------------- 
plt.figure(figsize=(14,6)) 
plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=2, alpha=0.7) 
plt.plot(pred.time_index, p, label="SVR Predicted", lw=2, color="red", linestyle="--") 
plt.title(f"{spi_column} SVR Forecast vs Actual") 
plt.xlabel("Date") 
plt.ylabel(spi_column) 
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6) # drought line 
plt.legend() 
plt.grid(True) 
plt.show()


# -----------------------------
# Forecast till 2099 (multi-step chunks)
# -----------------------------
last_date = df['ds'].max()
months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)

step_size = 12
forecast_series = series
all_forecasts = []

# while len(all_forecasts) < months_to_2099:
forecast_end_date = pd.Timestamp("2099-12-31")
while forecast_series.end_time() < forecast_end_date:
    step = min(step_size, months_to_2099 - len(all_forecasts))
    f = best_model.predict(n=step, series=forecast_series)
    all_forecasts.append(f)
    forecast_series = forecast_series.append(f)

# merge forecasts
forecast = all_forecasts[0]
for f in all_forecasts[1:]:
    forecast = forecast.append(f)

forecast_values = scaler.inverse_transform(forecast.values())
plt.figure(figsize=(16,6)) 
plt.plot(df['ds'], df[spi_column], label="Original SPI", lw=2, alpha=0.7) 
plt.plot(forecast.time_index, forecast_values, label="SVR Forecast", lw=2, color="green", linestyle="--") 
plt.title(f"{spi_column} SVR Forecast till 2099") 
plt.xlabel("Date") 
plt.ylabel(spi_column) 
plt.axhline(-1.5, color='black', linestyle='--', alpha=0.6) 
plt.legend() 
plt.grid(True) 
plt.show()
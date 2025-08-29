import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import RegressionModel, RNNModel
from darts.metrics import rmse
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

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
num_epochs = 300
input_folder = "./Data/testdata"
output_folder = "./results/main"
os.makedirs(output_folder, exist_ok=True)

def run_model_and_forecast(df, spi_column, station_name, model_name):
    results = {}

    # -----------------------------
    # Decide if we need scaling
    # -----------------------------
    use_scaler = model_name in ["SVR", "LSTM"]

    if use_scaler:
        scaler = StandardScaler()
        df[spi_column + "_scaled"] = scaler.fit_transform(df[[spi_column]])
        value_col = spi_column + "_scaled"
    else:
        scaler = None
        value_col = spi_column

    series = TimeSeries.from_dataframe(df, "ds", value_col)
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
            output_chunk_length=horizon,
            verbose=True
        )

    elif model_name == "RandomForest":
        model = RegressionModel(
            model=RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
            ),
            lags=window_size,
            output_chunk_length=horizon,
            verbose=True
        )

    elif model_name == "SVR":
        model = RegressionModel(
            model=SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
            lags=window_size,
            output_chunk_length=horizon,
            verbose=True
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
            random_state=SEED,
            verbose=True
        )

    else:
        raise ValueError("Unknown model")

    # -----------------------------
    # Train + Predict
    # -----------------------------
    model.fit(train)
    pred = model.predict(len(test), series=train)

    # Inverse transform if scaled
    if use_scaler:
        o = scaler.inverse_transform(test.values().reshape(-1, 1)).flatten()
        p = scaler.inverse_transform(pred.values().reshape(-1, 1)).flatten()
    else:
        o = test.values().flatten()
        p = pred.values().flatten()

    rmse_val = rmse(test, pred)
    corr_val = pearsonr(o, p)[0]

    # -----------------------------
    # Forecast till 2099
    # -----------------------------
    last_date = df["ds"].max()
    months_to_2099 = (2099 - last_date.year) * 12 + (12 - last_date.month + 1)
    forecast = model.predict(months_to_2099, series=series)

    if use_scaler:
        forecast_values = scaler.inverse_transform(forecast.values())
    else:
        forecast_values = forecast.values()

    # -----------------------------
    # Save Plot
    # -----------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df[spi_column], label="True")
    plt.plot(test.time_index, p, label="Test Prediction")
    plt.plot(forecast.time_index, forecast_values, label="Forecast till 2099")
    plt.legend()
    plt.title(f"{station_name} - {spi_column} - {model_name}")
    plt.savefig(
        os.path.join(output_folder, f"{station_name}_{spi_column}_{model_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    results["rmse"] = rmse_val
    results["corr"] = corr_val
    return results


# -----------------------------
# Main Loop: all stations + SPI columns
# -----------------------------
all_results = []

for file in glob.glob(os.path.join(input_folder, "*.csv")):
    station_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["ds"])
    spi_columns = [c for c in df.columns if c.startswith("SPI_")]

    for spi in spi_columns:
        for model_name in ["ExtraTrees", "RandomForest", "SVR", "LSTM"]:
            print(f"Running {model_name} on {station_name} - {spi}")
            try:
                res = run_model_and_forecast(df, spi, station_name, model_name)
                res.update({"station": station_name, "spi": spi, "model": model_name})
                all_results.append(res)
            except Exception as e:
                print(f"Error on {station_name}, {spi}, {model_name}: {e}")

# Save results summary
pd.DataFrame(all_results).to_csv(
    os.path.join(output_folder, "summary_metrics.csv"), index=False
)
print("✅ Done! Results saved in:", output_folder)

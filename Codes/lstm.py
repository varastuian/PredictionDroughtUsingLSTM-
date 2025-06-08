import pandas as pd
import numpy as np
import torch
from scipy.stats import gamma, norm
from darts.models import RNNModel
from darts.metrics import smape
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from darts.dataprocessing.transformers.scaler import Scaler
from darts import TimeSeries
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ----- Helper Functions -----
def fit_gamma(precip: pd.Series) -> tuple:
    non_zero = precip[precip > 0]
    if len(non_zero) == 0:
        return np.nan, np.nan, 1.0
    shape, _, scale = gamma.fit(non_zero, floc=0)
    zero_prob = (precip == 0).mean()
    return shape, scale, zero_prob

def compute_spi(roll: pd.Series, shape: float, scale: float, zero_prob: float) -> np.ndarray:
    probs = np.where(
        roll.isna(),
        np.nan,
        zero_prob + (1 - zero_prob) * gamma.cdf(roll, shape, scale=scale)
    )
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return norm.ppf(probs)

def process_station(df_station: pd.DataFrame) -> pd.DataFrame:
    df_station = df_station.sort_values('ds').reset_index(drop=True)
    result = {'ds': df_station['ds']}
    for s in [1, 3, 6, 9, 12, 24]:
        roll = df_station['precip'].rolling(s, min_periods=s).sum()
        shape, scale, zero_prob = fit_gamma(roll.dropna())
        spi = compute_spi(roll, shape, scale, zero_prob)
        mean, std = np.nanmean(spi), np.nanstd(spi)
        result[f'SPI_{s}'] = (spi - mean) / std
    result_df = pd.DataFrame(result)
    result_df['station_id'] = df_station['station_id'].iloc[0]
    return result_df


# ----- Load & Process Data -----
df = (
    pd.read_csv('main_data.csv', parse_dates=['data'])
    .assign(ds=lambda d: d['data'].dt.to_period('M').dt.to_timestamp())
    .groupby(['station_id', 'ds'])['rrr24']
    .sum()
    .reset_index(name='precip')
)

spi_list = [process_station(g) for _, g in df.groupby('station_id')]
all_spi = pd.concat(spi_list, ignore_index=True)
station_id = 40708
target_col = 'SPI_6'
horizon = 2
future_horizon = 360
window_size = 12
num_epochs = 200

df_station = all_spi[all_spi['station_id'] == station_id][['ds', target_col]].dropna()
series = TimeSeries.from_dataframe(df_station, time_col='ds', value_cols=target_col)


scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# ----- Step 1: Evaluation -----
train, val = series_scaled[:-horizon], series_scaled[-horizon:]

model = RNNModel(
    model='LSTM',
    input_chunk_length=window_size,
    output_chunk_length=horizon,
    hidden_dim=60,
    n_rnn_layers=5,
    dropout=0.05,
    batch_size=16,
    n_epochs=num_epochs,
    optimizer_kwargs={'lr': 1e-3},
    random_state=SEED
)


model.fit(train, verbose=True)
forecast = model.predict(horizon)

# Inverse transform
forecast = scaler.inverse_transform(forecast)
val_actual = scaler.inverse_transform(val)

# Evaluation
y_true = val_actual.values().flatten()
y_pred = forecast.values().flatten()

rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
mae_val = mean_absolute_error(y_true, y_pred)
r2_val = r2_score(y_true, y_pred)
corr_val, _ = pearsonr(y_true, y_pred)
smape_val = smape(val_actual, forecast)

print("Evaluation Metrics:")
print(f"RMSE:  {rmse_val:.3f}")
print(f"MAE:   {mae_val:.3f}")
print(f"R²:    {r2_val:.3f}")
print(f"Corr:  {corr_val:.3f}")
print(f"SMAPE: {smape_val:.2f}%")

# Plot predictions
series[-horizon:].plot(label='Actual')
forecast.plot(label='Forecast', lw=2)
plt.title(f"SPI_6 Forecast Evaluation - Station {station_id}")
plt.legend()
plt.show()

# ----- Step 2: Retrain on full series -----
model.fit(series_scaled, verbose=True)
future_forecast = model.predict(future_horizon)
future_forecast = scaler.inverse_transform(future_forecast)

# Plot future forecast
series.plot(label='Historical')
future_forecast.plot(label=f'Forecast (Next {future_horizon} Months)', lw=2)
plt.title(f"SPI_6 Forecast - {future_horizon} Months Ahead for Station {station_id}")
plt.legend()
plt.show()

# Optional: save forecast to CSV
# future_forecast.pd_dataframe().to_csv(f"SPI6_forecast_station_{station_id}.csv")

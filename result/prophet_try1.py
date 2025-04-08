import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from scipy.stats import norm
import numpy as np

# --- Step 1: Load data ---
df = pd.read_csv("result\merged_data.csv")  # replace with your actual file path

# --- Step 2: Preprocessing ---
df['data'] = pd.to_datetime(df['data'])
df['month'] = df['data'].dt.to_period('M')
df = df.sort_values(by=['station_id', 'data'])
print(df.iloc[3].to_string())
print(df.at[1, 'month'])  

# --- Step 3: Monthly precipitation per station ---
monthly_precip = df.groupby(['station_id', 'month'])['rrr24'].sum().reset_index()
monthly_precip['month'] = monthly_precip['month'].dt.to_timestamp()

# Rename columns for Prophet
monthly_precip.rename(columns={'month': 'ds', 'rrr24': 'precip'}, inplace=True)

# --- Step 4: Compute SPI function ---
def compute_spi(series, scale=12):
    """
    Compute Standardized Precipitation Index (SPI)
    """
    spi = pd.Series(index=series.index, dtype='float64')
    for i in range(scale, len(series)):
        window = series[i-scale:i]
        if window.std() != 0:
            spi.iloc[i] = (window.iloc[-1] - window.mean()) / window.std()
        else:
            spi.iloc[i] = 0
    return spi

# We'll demonstrate with one station
station_id = 40708  
station_data = monthly_precip[monthly_precip['station_id'] == station_id].copy()

# Compute SPI-12 (12 month scale)
station_data['spi'] = compute_spi(station_data['precip'], scale=12)

# Drop NaNs for modeling
spi_df = station_data.dropna(subset=['spi'])[['ds', 'spi']].copy()
spi_df.rename(columns={'spi': 'y'}, inplace=True)

# --- Step 5: Forecast using Prophet ---
model = Prophet()
model.fit(spi_df)

future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# --- Step 6: Plot SPI forecast ---
plt.figure(figsize=(12, 6))
plt.plot(spi_df['ds'], spi_df['y'], label='SPI (observed)')
plt.plot(forecast['ds'], forecast['yhat'], label='SPI Forecast')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.3)
plt.axhline(-1, color='orange', linestyle='--', label='Moderate Drought')
plt.axhline(-1.5, color='red', linestyle='--', label='Severe Drought')
plt.axhline(-2, color='purple', linestyle='--', label='Extreme Drought')
plt.title(f'SPI Forecast for Station {station_id}')
plt.xlabel('Date')
plt.ylabel('SPI Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

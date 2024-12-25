import pandas as pd
from scipy.stats import norm

# Load the data
data = pd.read_csv('Codes\merged_data.csv')

# Group data by station and calculate SPI
def calculate_spi(precipitation, scale=1):
    """
    Calculate Standardized Precipitation Index (SPI)
    """
    rolling_precip = precipitation.rolling(window=scale).mean()
    mean = rolling_precip.mean()
    std = rolling_precip.std()
    z_scores = (rolling_precip - mean) / std
    spi = norm.cdf(z_scores) * 2 - 1
    return spi

# Calculate SPI for each station
stations = data['station_id'].unique()
spi_results = []
for station in stations:
    station_data = data[data['station_id'] == station].sort_values('data')
    station_data['SPI'] = calculate_spi(station_data['rrr24'])
    spi_results.append(station_data)

# Combine results
spi_df = pd.concat(spi_results)

# Save to a new CSV
spi_df.to_csv('Codes\spi_results.csv', index=False)

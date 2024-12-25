import matplotlib.pyplot as plt
import pandas as pd

# Load SPI results
spi_df = pd.read_csv('Codes\spi_results.csv')

# Plot SPI trends for a specific station
def plot_spi(station_id):
    station_data = spi_df[spi_df['station_id'] == station_id]
    plt.figure(figsize=(12, 6))
    plt.plot(station_data['data'], station_data['SPI'], label=f'Station {station_id}')
    # plt.axhline(0, color='red', linestyle='--', label='Normal')
    # plt.axhline(-1, color='orange', linestyle='--', label='Moderate Drought')
    # plt.axhline(-2, color='brown', linestyle='--', label='Severe Drought')
    plt.xlabel('Month')
    plt.ylabel('SPI')
    plt.title(f'SPI Trend for Station {station_id}')
    plt.legend()
    plt.grid()
    plt.show()

# Example: Plot SPI for Station 1
# plot_spi(40708)

import geopandas as gpd

# Load geospatial data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
stations = spi_df.groupby('station_id').mean().reset_index()

# Create GeoDataFrame for stations
stations_gdf = gpd.GeoDataFrame(
    stations, geometry=gpd.points_from_xy(stations['lon'], stations['lat'])
)

# Plot map
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
world.plot(ax=ax, color='lightgrey')
stations_gdf.plot(
    ax=ax, 
    column='SPI', 
    cmap='coolwarm', 
    legend=True, 
    legend_kwds={'label': "SPI Index"}
)
plt.title("Station Drought Levels")
plt.show()


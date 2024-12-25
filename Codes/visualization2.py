import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the SPI results
spi_df = pd.read_csv('Codes\spi_results.csv')

# Plot SPI for a specific station
def plot_spi(station_id, spi_df):
    station_data = spi_df[spi_df['station_id'] == station_id]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='data', y='SPI', data=station_data, label='SPI')
    plt.axhline(0, color='red', linestyle='--', label='Drought Threshold')
    plt.title(f'SPI Trend for Station {station_id}')
    plt.xlabel('Month')
    plt.ylabel('SPI')
    plt.legend()
    plt.grid()
    plt.show()

# Example: Plot SPI for a specific station
plot_spi(station_id=40708, spi_df=spi_df)

def plot_drought_heatmap(spi_df):
    # pivot_table = spi_df.pivot('data', 'station_id', 'SPI')
    pivot_table = spi_df.pivot_table(index='data', columns='station_id', values='SPI')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='RdYlBu_r', center=0, annot=False, cbar_kws={'label': 'SPI'})
    plt.title('Drought Conditions Heatmap')
    plt.xlabel('Station ID')
    plt.ylabel('Month')
    plt.show()

# Generate heatmap
plot_drought_heatmap(spi_df)


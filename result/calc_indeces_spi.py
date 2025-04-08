import pandas as pd
import numpy as np
# from climate_indices import compute, indices, utils
from climate_indices.indices import spi
from climate_indices.compute import Periodicity, Distribution

# Load CSV
# df = pd.read_csv("your_data.csv", parse_dates=["data"])
# df = pd.read_csv("result\merged_data.csv", parse_dates=["data"])  
df = pd.read_csv("result\merged_data.csv", parse_dates=["data"], 
                 date_parser=lambda x: pd.to_datetime(x, format="%m/%d/%Y %I:%M:%S %p"))

# Filter for one station, e.g., Ardebil
station_df = df[df['station_name'] == "Ardebil"]

# Sort by date
station_df = station_df.sort_values("data")

# Get precipitation (assuming rrr24 is monthly total rainfall)
precip_mm = station_df['rrr24'].astype(float).values

# Handle missing data (e.g., fill with np.nan, mask it later)
precip_mm = np.ma.masked_invalid(precip_mm)

# Convert dates to year/month arrays
years = station_df["data"].dt.year.values
months = station_df["data"].dt.month.values

# Compute SPI at 1-month scale
# spi_values = indices.spi(
#     # precipitation=precip_mm,
#     scale=1,  # 1-month SPI
#     data_start_year=years[0],
#     calibration_year_initial=years[0],
#     calibration_year_final=years[-1]
#     # periodicity=utils.Periodicity.monthly
# )
spi_values = spi(
    values=precip_mm,                       # masked array of precipitation
    scale=1,                             # SPI scale (e.g., 1-month)
    distribution=Distribution.gamma,     # or Distribution.pearson
    data_start_year=years[0],
    calibration_year_initial=years[0],
    calibration_year_final=years[-1],
    periodicity=Periodicity.monthly
)


# Show result
print(spi_values)

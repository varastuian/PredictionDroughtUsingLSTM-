import matplotlib.pyplot as plt
import HandleData as HandleData
import seaborn as sns
import numpy as np
from scipy.stats import gamma, norm
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import xarray as xr
from scipy.stats import gamma, norm

df = pd.read_csv("Codes/merged_data.csv")
df['data'] = pd.to_datetime(df['data'])


station_df = df[df['station_id'] == 40708].copy()

station_df['year'] = station_df['data'].dt.year
station_df['month'] = station_df['data'].dt.month

# Aggregate to monthly precipitation totals using the 'rrr24' column (assumed precipitation)
monthly_precip = station_df.groupby(['year', 'month'])['rrr24'].sum().reset_index()

# Extract precipitation values from the monthly data
precip = monthly_precip['rrr24'].values

# Calculate the probability of zero precipitation
zero_prob = np.mean(precip == 0)

# Extract only the non-zero precipitation values for fitting the gamma distribution
nonzero_precip = precip[precip > 0]

# Fit a gamma distribution to the nonzero precipitation values (fixing the location parameter to 0)
fit_alpha, fit_loc, fit_beta = gamma.fit(nonzero_precip, floc=0)

# Define a function to compute SPI for a given precipitation value
def compute_spi(x):
    if x == 0:
        # Include the probability of zero precipitation
        prob = zero_prob
    else:
        prob = zero_prob + (1 - zero_prob) * gamma.cdf(x, a=fit_alpha, loc=fit_loc, scale=fit_beta)
    # Transform cumulative probability to a z-score (SPI value)
    return norm.ppf(prob)

# Apply the function to calculate SPI for each monthly total
monthly_precip['SPI'] = monthly_precip['rrr24'].apply(compute_spi)

# Display the resulting SPI values along with the corresponding year and month
print(monthly_precip)
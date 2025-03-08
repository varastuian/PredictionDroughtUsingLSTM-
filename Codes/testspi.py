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


station_df = pd.read_csv(r".\result\40708.csv")
precip = station_df['rrr24'].values
zero_prob = np.mean(precip == 0)
nonzero_precip = precip[precip > 0]

# fit_alpha, fit_loc, fit_beta = gamma.fit(nonzero_precip, floc=0)

def compute_spi(x):
    # if x == 0:
    #     prob = zero_prob
    # else:
    #     prob = zero_prob + (1 - zero_prob) * gamma.cdf(x, a=fit_alpha, loc=fit_loc, scale=fit_beta)
    # return norm.ppf(prob)
    x = x[x > 0]
    shape, loc, scale =gamma.fit(x, floc=0)
    cdf = gamma.cdf(x, shape, loc, scale)
    spi_values = norm.ppf(cdf)
    return spi_values

def compute_spi_timescales(data, timescales):
    spi_results = {}
    
    for scale in timescales:
        rolling_precip = data['rrr24'].rolling(scale, min_periods=scale).sum()
        spi_results[f"SPI_{scale}"] = compute_spi(rolling_precip.dropna())
    return spi_results


with open('./result/withspi.txt', 'w') as f:
        
    timescales = [1, 3, 6,9,12,24]  
    spi_results = compute_spi_timescales(station_df, timescales)
    
    for i, (_, row) in enumerate(station_df.iterrows()):
        date_str = f"{int(row['month'])}/1/{int(row['year'])}"
        spi_values = " ".join([f"SPI_{scale}: {spi_results[f'SPI_{scale}'][i]:.2f}" if i >= scale - 1 else f"SPI_{scale}: NA" for scale in timescales])
        f.write(f"{date_str} {row['rrr24']} {spi_values}\n")
# station_df['SPI'] = station_df['rrr24'].apply(compute_spi)
# station_df.to_csv('./result/withspi.csv', index=False)


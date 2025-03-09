import numpy as np
from scipy.stats import gamma, norm
import pandas as pd
from scipy.stats import gamma, norm


station_df = pd.read_csv(r".\result\40708.csv")
precip = station_df['rrr24'].values
zero_prob = np.mean(precip == 0)

def compute_spi(x):
    nonzero = x[x > 0]
    shape, loc, scale_param = gamma.fit(nonzero, floc=0)
    
    spi_values = np.empty_like(x, dtype=float)
    for i, val in enumerate(x):
        if val == 0:
            # For zero precipitation, use the zero probability directly.
            prob = zero_prob
        else:
            # Compute the cumulative probability
            prob = zero_prob + (1 - zero_prob) * gamma.cdf(val, shape, loc, scale_param)
        spi_values[i] = norm.ppf(prob)
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
        # spi_values = " ".join([f"SPI_{scale}: {spi_results[f'SPI_{scale}'][i]:.2f}" if i >= scale - 1 else f"SPI_{scale}: NA" for scale in timescales])
        spi_values_list = []
        for scale in timescales:
            if i >= scale - 1:
                spi_index = i - (scale - 1)  # adjust for the dropped rows
                spi_val = spi_results[f"SPI_{scale}"][spi_index]
                spi_values_list.append(f"SPI_{scale}: {spi_val:.2f}")
            else:
                spi_values_list.append(f"SPI_{scale}: NA")
        spi_values = " ".join(spi_values_list)
        f.write(f"{date_str} {row['rrr24']} {spi_values}\n")
# station_df['SPI'] = station_df['rrr24'].apply(compute_spi)
# station_df.to_csv('./result/withspi.csv', index=False)


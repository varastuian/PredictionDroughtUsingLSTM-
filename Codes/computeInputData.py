import os
import pandas as pd
import numpy as np
from scipy.stats import gamma, norm


def compute_spi_per_month(df: pd.DataFrame, scale: int) -> pd.Series:
    """
    Compute SPI for a given time scale, month by month.
    Returns a pandas Series aligned with df['ds'].
    """
    spi_vals = pd.Series(np.nan, index=df.index, dtype=float)

    # rolling total over 'scale' months
    roll = df['precip'].rolling(scale, min_periods=scale).sum()

    for m in range(1, 13):  # fit distribution separately for each calendar month
        mask = df['ds'].dt.month == m
        vals = roll[mask]

        if vals.isna().all():
            continue

        nonzero = vals[vals > 0]
        if len(nonzero) < 5:  # too little data to fit reliably
            continue

        # fit gamma (loc=0)
        shape, _, scale_param = gamma.fit(nonzero, floc=0)
        zero_prob = (vals == 0).mean()

        # convert to cumulative probability
        probs = zero_prob + (1 - zero_prob) * gamma.cdf(vals, shape, scale=scale_param)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)  # numerical safety

        # normal transform → SPI
        spi_vals.loc[mask] = norm.ppf(probs)

    return spi_vals


def process_station(df_station: pd.DataFrame) -> pd.DataFrame:
    """
    Process one station and compute SPI for 1,3,6,9,12,24 months.
    Returns a DataFrame with ds, precip, temperature, and SPI columns.
    """
    df_station = df_station.sort_values('ds').reset_index(drop=True)
    result = df_station[['station_id', 'ds', 'precip', 'tm_m']].copy()

    for s in [1, 3, 6, 9, 12, 24]:
        result[f'SPI_{s}'] = compute_spi_per_month(df_station, s)

    return result


# =====================
# MAIN EXECUTION
# =====================

# Load & aggregate daily rainfall to monthly totals (keeping tm_m too)
df = (
    pd.read_csv('./Data/raw_data.csv', parse_dates=['data'])
      .assign(ds=lambda d: d['data'].dt.to_period('M').dt.to_timestamp())
      .groupby(['station_id', 'ds'], as_index=False)
      .agg(precip=('rrr24', 'sum'),
           tm_m=('tm_m', 'mean'))  # keep monthly mean temperature
)

# Output folder
output_dir = "./Data/python_spi2"
os.makedirs(output_dir, exist_ok=True)

# Loop through stations, compute SPI, save CSV
for station_id, group in df.groupby("station_id"):
    spi_df = process_station(group)

    filename = os.path.join(output_dir, f"{station_id}.csv")
    spi_df.to_csv(filename, index=False, date_format="%m/%d/%Y")

print(f"✅ Saved {df['station_id'].nunique()} station SPI files in {output_dir}")

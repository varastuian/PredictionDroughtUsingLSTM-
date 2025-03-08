# import HandleData as hd

# df = hd.get_data('Codes/spi_results.csv')
# hd.by_station(40708)


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
import os

print(os.getcwd())

station_df = pd.read_csv(r"..\result\40708.csv")


# station_df['monthly_precip'] = station_df.groupby('year')['rrr24'].sum().reset_index()
# station_df.to_csv('../result/gbyyear.csv', index=False)

    
# Group data by station_id
grouped = station_df.groupby('station_id')

with open(r"..\result\newformat.txt", 'w') as f:
    for station_id, data in grouped:
        f.write(f"{station_id}\n")  # Write station ID
        
        for _, row in data.iterrows():
            date_str = f"{int(row['month'])}/1/{int(row['year'])}"
            f.write(f"{date_str} {row['rrr24']}\n")  # Write formatted date and value


import pandas as pd
import os
df = pd.DataFrame()

def get_data(path):
    global df
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['data'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df


def show_data():
    df.head()
    stations = df[['station_name', 'station_id']].drop_duplicates()
    print(stations)

def by_station(stations_id):
    station_df = df[df['station_id'] == stations_id]
    station_df = station_df[['year', 'month', 'station_id', 'rrr24']]
    station_df = station_df.sort_values(by=['year', 'month'], ascending=True)

    # station_df = station_df.drop(columns=[col for col in df.columns if col not in ['station_id', 'year','month', 'rrr24']])
    os.makedirs('result',exist_ok=True)
    station_df.to_csv(f'result/{stations_id}.csv', index=False)


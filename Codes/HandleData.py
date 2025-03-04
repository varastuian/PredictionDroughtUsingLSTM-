import pandas as pd

df = pd.DataFrame()

def get_data():
    global df
    df = pd.read_csv('spi_results.csv')
    df['date'] = pd.to_datetime(df['data'])
    return df


def show_data():
    df.head()
    stations = df[['station_name', 'station_id']].drop_duplicates()
    print(stations)

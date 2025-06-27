import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pywt
import matplotlib.pyplot as plt
import os


def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.rename(columns={
        "ds": "date",
        "precip": "rainfall",
        "SPI_1": "spi1",
        "SPI_3": "spi3",
        "SPI_6": "spi6",
        "SPI_9": "spi9",
        "SPI_12": "spi12",
        "SPI_24": "spi24",
        "station_id": "station"
    }, inplace=True)
    df.set_index("date", inplace=True)
    # spi_dfs = {}
    # for col in ["spi1", "spi3", "spi6", "spi9", "spi12", "spi24"]:
    #     if col in df.columns:
    #         sub_df = df[["rainfall", col, "station"]].dropna()
    #         sub_df = sub_df.rename(columns={col: "spi"})
    #         spi_dfs[col] = sub_df

    return df


def remove_seasonality(series):
    # Remove monthly seasonality via decomposition
    decomposition = seasonal_decompose(series, model='additive', period=12, extrapolate_trend='freq')
    return series - decomposition.seasonal


def make_features(df, timescale, lags=12):
    # Use SPI timescale and rainfall as covariates
    data = df[[f"spi{timescale}", "rainfall"]].copy()
    data = data.rename(columns={f"spi{timescale}": "spi"})
    data.dropna(inplace=True)

    # remove seasonality
    data['spi_deseason'] = remove_seasonality(data['spi'])
    # lag features
    for lag in range(1, lags+1):
        data[f'spi_lag_{lag}'] = data['spi_deseason'].shift(lag)
    data.dropna(inplace=True)
    return data


def train_models(X_train, y_train):
    models = {
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }

    # Fit tree-based and SVR
    for name, model in models.items():
        model.fit(X_train, y_train)
        models[name] = model

    # Prepare data for LSTM (3D)
    def build_lstm(X, y):
        X3 = X.values.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential([
            LSTM(64, input_shape=(X3.shape[1], X3.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X3, y, epochs=50, batch_size=32, verbose=0)
        return model

    models['LSTM'] = build_lstm(X_train, y_train)

    # Wavelet-based LSTM (decompose then LSTM)
    def build_wblstm(X, y):
        # Discrete Wavelet Transform on each feature column
        coeffs = [pywt.wavedec(X.iloc[:, i], 'db1', level=2) for i in range(X.shape[1])]
        # Use approximation coefficients at level 2
        arrs = [c[0] for c in coeffs]
        X_wav = np.vstack(arrs).T
        X3 = X_wav.reshape(X_wav.shape[0], 1, X_wav.shape[1])
        model = Sequential([
            LSTM(64, input_shape=(1, X_wav.shape[1])),
            Dropout(0.2),
            Dense(1)
        ])
        y = y[-X3.shape[0]:]

        model.compile(optimizer='adam', loss='mse')
        model.fit(X3, y, epochs=50, batch_size=32, verbose=0)
        return model

    models['WB-LSTM'] = build_wblstm(X_train, y_train)

    return models


def evaluate(model, X, y, model_type):
    if model_type in ['LSTM', 'WB-LSTM']:
        X3 = X.values.reshape(X.shape[0], 1, X.shape[1])
        y_pred = model.predict(X3).flatten()
    else:
        y_pred = model.predict(X)
    return mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred), y_pred


def forecast(model, X, model_type):
    if model_type in ['LSTM', 'WB-LSTM']:
        X3 = X.values.reshape(X.shape[0], 1, X.shape[1])
        return model.predict(X3).flatten()
    else:
        return model.predict(X)

def plot_all_model_predictions(y_true, all_preds, all_metrics, station, timescale):
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth
    plt.plot(y_true.index, y_true, label='Validation SPI', linewidth=2, color='black')

    # Plot each model's prediction
    for model_name, y_pred in all_preds.items():
        plt.plot(y_true.index, y_pred, linestyle='--', label=f'{model_name} (RMSE={all_metrics[model_name]["rmse"]:.2f})')

    plt.title(f'Model Comparison - Station {station}, SPI{timescale}')
    plt.xlabel('Date')
    plt.ylabel('SPI')
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/all_predictions_{station}_spi{timescale}.png')
    plt.close()


def plot_predictions(y_true, y_pred, station, timescale, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.index, y_true, label='Validation SPI')
    plt.plot(y_true.index, y_pred, label='Predicted SPI', linestyle='--')
    plt.title(f'{model_name} Prediction vs Validation - Station {station}, SPI{timescale}')
    plt.xlabel('Date')
    plt.ylabel('SPI')
    plt.legend()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/prediction_{station}_spi{timescale}_{model_name}.png')
    plt.close()


def run_pipeline(csv_path):
    df = load_data(csv_path)
    stations = df['station'].unique()
    results = []

    for st in [40700]:
        df_st = df[df['station'] == st].copy()
        for ts in [6]:
            data = make_features(df_st, ts)
            # train-test split (80/20)
            split = int(len(data)*0.8)
            X = data.drop(columns=['spi', 'rainfall'])
            y = data['spi_deseason']
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            models = train_models(X_train, y_train)
            # evaluate all
            metrics = {}
            preds = {}
            for name, model in models.items():
                rmse, mae, y_pred = evaluate(model, X_val, y_val, name)
                metrics[name] = {'rmse': rmse, 'mae': mae}
                preds[name] = y_pred

            # select best by rmse
            best = min(metrics, key=lambda m: metrics[m]['rmse'])
            results.append({'station': st, 'timescale': ts, 'best_model': best, 'metrics': metrics[best]})

            # forecast next (last available row)
            next_X = X.iloc[[-1]]
            fcast = forecast(models[best], next_X, best)[0]
            results[-1]['forecast'] = fcast

            # plot prediction vs validation
            plot_all_model_predictions(y_val, preds, metrics, st, ts)

            plot_predictions(y_val, preds[best], st, ts, best)

    return pd.DataFrame(results)


if __name__ == '__main__':
    csv_path = Path('codes/finaldata.csv')
    _ = run_pipeline(csv_path)
    # summary.to_csv('drought_forecast_results.csv', index=False)
    print("Pipeline complete.")

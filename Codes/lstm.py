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
import matplotlib.pyplot as plt
from skill_metrics import taylor_diagram
import seaborn as sns
os.makedirs('plots', exist_ok=True)

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

    return df


def remove_seasonality(series):
    decomposition = seasonal_decompose(series, model='additive', period=12, extrapolate_trend='freq')
    deseasonalized = series - decomposition.seasonal
    return deseasonalized, decomposition.seasonal


def make_features(df, timescale, lags=12):
    # Use SPI timescale and rainfall as covariates
    data = df[[f"spi{timescale}", "rainfall"]].copy()
    data = data.rename(columns={f"spi{timescale}": "spi"})
    data.dropna(inplace=True)



    # remove seasonality
    # data['spi_deseason'] = remove_seasonality(data['spi'])
    data['spi_deseason'], seasonal = remove_seasonality(data['spi'])
    data['seasonal'] = seasonal

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
    
def forecast_until_now(best_model, model_type, X_last, seasonal_vals, horizon):
    preds = []
    X_curr = X_last.copy()
    seasonal_vals = list(seasonal_vals[-12:])  # repeat pattern if needed

    for i in range(horizon):
        if model_type in ['LSTM', 'WB-LSTM']:
            X_input = X_curr.values.reshape(1, 1, -1)
        else:
            X_input = X_curr.values.reshape(1, -1)

        yhat_deseason = best_model.predict(X_input).flatten()[0]
        yhat_full = yhat_deseason + seasonal_vals[i % 12]
        preds.append(yhat_full)

        # Shift features: drop oldest lag, append new value
        # next_row = X_curr.values.flatten()[1:-1].tolist() + [yhat_deseason]
        # X_curr = pd.DataFrame([next_row], columns=X_curr.columns, index=[X_curr.index[0] + pd.DateOffset(months=1)])
        # Roll lag features left and add new prediction at the end
        next_vals = X_curr.iloc[0, 1:].tolist() + [yhat_deseason]
        X_curr = pd.DataFrame([next_vals], columns=X_curr.columns, index=[X_curr.index[0] + pd.DateOffset(months=1)])


    forecast_dates = pd.date_range(start=X_last.index[0] + pd.DateOffset(months=1), periods=horizon, freq='MS')
    return pd.Series(preds, index=forecast_dates, name='forecast_spi')

    

def plot_all_model_predictions_with_spi(y_val_deseason, all_preds_deseason, seasonal_vals, station, timescale):
    # Reconstruct true full SPI
    y_val_spi = y_val_deseason + seasonal_vals

    # Reconstruct predicted full SPI for each model
    plt.figure(figsize=(12, 6))
    plt.plot(y_val_spi.index, y_val_spi, label='Actual SPI', linewidth=2, color='black')

    for model_name, y_pred_deseason in all_preds_deseason.items():
        y_pred_spi = y_pred_deseason + seasonal_vals
        plt.plot(y_val_spi.index, y_pred_spi, linestyle='--', label=model_name)

    plt.title(f'All Models - Predicted vs Actual SPI\nStation {station}, SPI{timescale}')
    plt.xlabel('Date')
    plt.ylabel('SPI')
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/all_models_pred_vs_actual_spi_{station}_spi{timescale}.png')
    plt.close()


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



class TaylorDiagram:
    def __init__(self, ref_std, fig=None, rect=111, label='Reference'):
        self.ref_std = ref_std

        self.fig = fig or plt.figure()
        self.ax = self.fig.add_subplot(rect, polar=True)

        # Plot reference point and standard deviation arc
        t = np.linspace(0, np.pi/2)
        self.ax.plot(t, [ref_std]*len(t), 'k--', label=label)
        self.ax.set_xlim(0, np.pi/2)
        self.ax.set_ylim(0, ref_std*1.5)
        self.ax.set_xlabel('Correlation')
        self.ax.set_ylabel('Standard Deviation')

    def add_sample(self, stddev, corrcoef, label='', marker='o', c='b'):
        angle = np.arccos(corrcoef)
        self.ax.plot(angle, stddev, marker, label=label, color=c)

def run_pipeline(csv_path):
    df = load_data(csv_path)
    stations = df['station'].unique()
    results = []
    summary = []

    for st in [40700]:
    # for st in stations:
        df_st = df[df['station'] == st].copy()
        for ts in [3]:
        # for ts in [1,3,6,9,12,24]:
            data = make_features(df_st, ts)
            # train-test split (80/20)
            split = int(len(data)*0.8)

            train_data = data.iloc[:split]
            val_data = data.iloc[split:]



            # plt.figure(figsize=(12, 6))

            # # Plot training SPI
            # plt.plot(train_data.index, train_data['spi'], label='Training SPI', color='blue')

            # # Plot validation SPI
            # plt.plot(val_data.index, val_data['spi'], label='Validation SPI', color='orange')

            # # Optional: vertical line at split
            # plt.axvline(val_data.index[0], color='gray', linestyle='--', label='Train/Val Split')

            # plt.title(f'SPI{ts} Time Series (Train vs Validation)')
            # plt.xlabel('Date')
            # plt.ylabel('SPI')
            # plt.legend()
            # plt.grid(True)
            # os.makedirs('plots', exist_ok=True)
            # plt.savefig(f'plots/spi{ts}_train_val_split_station_{st}.png')
            # plt.close()


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


            for name in models:
                rmse, mae = metrics[name]['rmse'], metrics[name]['mae']
                corr = np.corrcoef(y_val, preds[name])[0,1]
                std_e = np.std(preds[name] - y_val)
                std_o = np.std(y_val)
                summary.append({'timescale': ts, 'model': name, 'RMSE': rmse, 'MAE': mae, 'Corr': corr, 'Std_err': std_e, 'Std_obs': std_o})


             

            ref_std = np.std(y_val)
            fig = plt.figure(figsize=(6, 6))
            dia = TaylorDiagram(ref_std, fig=fig, label='Observed')

            colors = ['r', 'g', 'b', 'm', 'c']
            for i, (name, y_pred) in enumerate(preds.items()):
                std = np.std(y_pred)
                corr = np.corrcoef(y_val, y_pred)[0, 1]
                dia.add_sample(std, corr, label=name, c=colors[i % len(colors)])

            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.title(f'Taylor Diagram - Station {st}, SPI{ts}')
            plt.savefig(f'plots/taylor_diagram_station_{st}_spi{ts}.png')
            plt.close()


            # forecast 
            last_date = data.index[-1]
            current_date = pd.to_datetime('today').normalize()
            horizon = (current_date.year - last_date.year) * 12 + (current_date.month - last_date.month)

            # next_X = X.iloc[[-1]]
            # fcast = forecast(models[best], next_X, best)[0]
            # results[-1]['forecast'] = fcast
            seasonal_vals = data['seasonal'].iloc[-12:]  # last year’s seasonality
            X_last = X.iloc[[-1]]
            best_model = models[best]
            model_type = best

            plot_start = pd.to_datetime('2024-01-01')

            # Slice historical SPI from 2024 onward
            hist_spi = data.loc[plot_start:, 'spi']

            pred_series = forecast_until_now(best_model, model_type, X_last, seasonal_vals, horizon)
            pred_series.to_csv(f'forecast_until_now_station_{st}_spi{ts}.csv')

            plt.figure(figsize=(12, 5))
            plt.plot(hist_spi.index, hist_spi, label='Historical SPI (2024–mid‑2024)', color='blue')

            # plt.plot(data.index, data['spi'], label='Historical SPI', color='blue')
            plt.plot(pred_series.index, pred_series, label='Forecasted SPI', color='red', linestyle='--')
            plt.axvline(data.index[-1], color='gray', linestyle='--', label='Forecast Start')
            plt.title(f'SPI Forecast from {data.index[-1].strftime("%Y-%m")} to {pred_series.index[-1].strftime("%Y-%m")}')
            plt.xlabel('Date')
            plt.ylabel('SPI')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plots/spi_forecast_until_now_station_{st}_spi{ts}.png')
            plt.close()




            #heatmap
            dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=10, freq='MS')
            forecast_10 = {}
            for name, model in models.items():
                forecast_10[name] = forecast_until_now(model, name, X_val.iloc[[-1]], data['seasonal'].iloc[split:], horizon=10)

            heat = pd.DataFrame(forecast_10, index=dates)

            # heat = pd.DataFrame({name: preds[name] for name in models}, index=dates)
            best = heat.idxmin(axis=1)  # lowest RMSE per month?

            codes = {name: i for i, name in enumerate(models)}
            heat_codes = best.map(codes)

            plt.figure(figsize=(10,2))
            sns.heatmap(heat_codes.values.reshape(1,-1), cmap='tab10', cbar=False,
                        xticklabels=dates.strftime('%Y-%m'), yticklabels=['Best model'])
            plt.savefig('forecast_best_model_heatmap.png')


            seasonal_vals = data['seasonal'].iloc[split:]  # seasonal component for validation period
            # plot_predicted_vs_actual_spi(y_val, preds[best], seasonal_vals, st, ts, best)
            plot_all_model_predictions_with_spi(y_val, preds, seasonal_vals, st, ts)


            # plot prediction vs validation
            plot_all_model_predictions(y_val, preds, metrics, st, ts)

            # plot_predictions(y_val, preds[best], st, ts, best)

    metrics_df = pd.DataFrame(summary)
    metrics_df.to_csv('metrics_summary.csv', index=False)
    return pd.DataFrame(results)


if __name__ == '__main__':
    csv_path = Path('codes/finaldata.csv')
    _ = run_pipeline(csv_path)
    # summary.to_csv('drought_forecast_results.csv', index=False)
    print("Pipeline complete.")

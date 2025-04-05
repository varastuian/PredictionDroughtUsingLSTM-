import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import time


def forcast(processed_series, selected_lags, lstm_model):
    n_forecast = 120
    forecast = []

    base_series = processed_series
    current_input = base_series[-selected_lags:].tolist()  # length = max(selected_lags)
    for i in range(n_forecast):
        X_input = np.array(current_input).astype('float32')
        X_tensor = torch.tensor(X_input.reshape(1, len(current_input), 1), dtype=torch.float32)
        with torch.no_grad():
            next_val = lstm_model(X_tensor).item()
        forecast.append(next_val)
        current_input.pop(0)
        current_input.append(next_val)

# Plot the forecast along with the historical series.
    plt.figure(figsize=(10,5))
    plt.plot(range(len(base_series)), base_series, label='Historical Series')
    plt.plot(range(len(base_series), len(base_series)+n_forecast), forecast, label=f'{n_forecast} month Forecast', color='red')
    plt.title(f'{n_forecast} Month Forecast using lstm')
    plt.xlabel('Time Index (months)')
    plt.ylabel('SPI')
    plt.legend()

    plt.savefig(f"C:\\Users\\varas\\Desktop\\fig{int(time.time())}")


def create_dataset(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)



class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # h0 = torch.zeros(1, x.size(0), 50)
        # c0 = torch.zeros(1, x.size(0), 50)

        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out



if __name__=="__main__":
        
    ######################################
    # 1. Read and Preprocess Data
    ######################################
    df = pd.read_csv('result/40708spi.txt', delimiter=' ')
    series = df['spi1'].values.astype('float32')


    ######################################
    # 2. Wavelet Transformation (DWT)
    ######################################
    wavelet = 'db4'
    level = 2
    coeffs = pywt.wavedec(series, wavelet, level=level)
    # Approximation: a denoised/smoothed version
    approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(series))
    # Detail from level 1 (used for boosted methods)
    detail = pywt.upcoef('d', coeffs[1], wavelet, level=level, take=len(series))

    # plt.figure(figsize=(10,4))
    # plt.plot(series, label='Original SPI', alpha=0.6)
    # plt.plot(approx, label='Approximation', linewidth=2)
    # plt.plot(detail, label='Detail (Level 1)', linestyle='--')
    # plt.title('SPI with Wavelet Components')
    # plt.xlabel('Time index')
    # plt.legend()
    # # plt.show()

    # For original models, we use the approximation as the processed series.
    processed_series = approx

    ######################################
    # 3. Lag Selection using ACF/PACF
    ######################################

    selected_lags = 12

    ######################################
    # 4. Create Dataset
    ######################################
    torch.manual_seed(42)

    # Dataset for "approximation-based" methods:
    X_all, y_all = create_dataset(processed_series, selected_lags)
    train_size = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    ######################################
    # 6. Model 1: LSTM (Using approx dataset)
    ######################################

    lstm_model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
    n_epochs = 100
    lstm_model.train()
    for epoch in range(n_epochs):
        output = lstm_model(X_train.unsqueeze(-1)).squeeze()  # Add .squeeze() here

        optimizer.zero_grad()
        # output = lstm_model(X_batch)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}")

    
    for i in range(10):        
        forcast(processed_series, selected_lags, lstm_model)
        print(i)



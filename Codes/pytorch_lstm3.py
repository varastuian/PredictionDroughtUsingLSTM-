import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pywt
import time

def lstm_predict():
    torch.manual_seed(42)

    # Load the data
    df = pd.read_csv('result/40708spi.txt', delimiter=' ')
    df['date'] = pd.to_datetime(df['date'],dayfirst=False)  # Assuming there's a 'date' column
    history = df['spi1'].values.astype('float')

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length - 1):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    lags = [12, 24, 36]
    forecast_all = []

    for lag in lags:
        # Create sequences for the current lag
        seq_length = lag
        X, y = create_sequences(history, seq_length)

        # Split the data into training and testing sets
        train_size = int(len(y) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        # Initialize the LSTM model
        input_size = 1
        hidden_size = 50
        num_layers = 1
        output_size = 1
        model = LSTM(input_size, hidden_size, num_layers, output_size)

        # Set training parameters
        learning_rate = 0.01
        num_epochs = 100

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            outputs = model(X_train.unsqueeze(-1)).squeeze()  # Add .squeeze() here
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

        # Forecasting
        n_forecast = 120
        forecast = []
        base_series = history
        current_input = base_series[-seq_length:].tolist()
        for i in range(n_forecast):
            X_input = np.array(current_input).astype('float32')
            X_tensor = torch.tensor(X_input.reshape(1, len(current_input), 1), dtype=torch.float32)
            with torch.no_grad():
                next_val = model(X_tensor).item()
            forecast.append(next_val)
            current_input.pop(0)
            current_input.append(next_val)

        # Plot the forecasted part only with dates
        plt.figure(figsize=(10, 5))
        forecast_dates = pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='M')
        plt.plot(forecast_dates, forecast, label=f'{n_forecast} month Forecast (Lag {lag})', color='red')
        plt.title(f'{n_forecast} Month Forecast using LSTM (Lag {lag})')
        plt.xlabel('Date')
        plt.ylabel('SPI')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"C:\\Users\\varas\\Desktop\\fig_lag{lag}_{int(time.time())}.png")

if __name__ == "__main__":
    # for i in range(5):
        lstm_predict()



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pywt
import time

def lstm_predict():
    torch.manual_seed(42)


    df = pd.read_csv('result/40708spi.txt', delimiter=' ')
    history = df['spi1'].values.astype('float')
# history = history.astype(float)
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length - 1):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

# Preprocess the historical data
    seq_length = 24
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
        # out, _ = self.lstm(x)
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


# with torch.no_grad():
#     test_outputs = model(X_test.unsqueeze(-1)).squeeze()  # Add .squeeze() here
#     test_loss = criterion(test_outputs, y_test)
#     print(f"Test Loss: {test_loss.item():.4f}")



# Concatenate the training and test predictions
# with torch.no_grad():
#     train_outputs = model(X_train.unsqueeze(-1)).squeeze().numpy()
#     test_outputs = model(X_test.unsqueeze(-1)).squeeze().numpy()
# all_outputs = np.concatenate((train_outputs, test_outputs))

# # Calculate the index where the test set starts
# test_start_index = len(history) - len(y_test) - seq_length

# Plot the true values and the predictions
# plt.plot(history, label="True Values")
# plt.plot(range(seq_length, seq_length + len(all_outputs)), all_outputs, label="Predictions")
# plt.axvline(x=test_start_index, color='gray', linestyle='--', label="Test set start")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.legend()
# plt.title("LSTM Predictions vs True Values")
# plt.show()



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
    plt.figure(figsize=(10,5))

    plt.plot(range(len(base_series)), base_series, label='Historical Series')
    plt.plot(range(len(base_series), len(base_series)+n_forecast), forecast, label=f'{n_forecast} month Forecast', color='red')
    plt.title(f'{n_forecast} Month Forecast using lstm')
    plt.xlabel('Time Index (months)')
    plt.ylabel('SPI')
    plt.legend()
# plt.show()
    plt.savefig(f"C:\\Users\\varas\\Desktop\\fig{int(time.time())}")

if __name__=="__main__":
    for i in range(5):        
        lstm_predict()
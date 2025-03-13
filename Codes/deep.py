import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ---------------- Data Loading & Preprocessing ----------------

def load_data(file_path, column='spi1', date_format='%d/%m/%Y', delimiter=' '):
    """
    Load and preprocess the data.
    - Reads the file with the given delimiter.
    - Parses the date column.
    - Sets the date as index.
    - Replaces missing values (-99) and fills them (using forward fill).
    """
    data = pd.read_csv(file_path, delimiter=delimiter)
    data['date'] = pd.to_datetime(data['date'], format=date_format)
    data.set_index('date', inplace=True)
    # Replace -99 with NaN and fill missing values (alternatively, you can use interpolation)
    data[column].replace(-99, np.nan, inplace=True)
    data[column].fillna(method='ffill', inplace=True)
    return data

def create_dataset(dataset, look_back=12):
    """
    Create sequences of data for LSTM training.
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(y)

# ---------------- Custom Dataset Class ----------------

class SPIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- LSTM Model Definition ----------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# ---------------- Training Function ----------------

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return losses

# ---------------- Forecasting Function ----------------

def forecast_future(model, last_sequence, scaler, forecast_steps=12, device='cpu'):
    """
    Iteratively predict future values.
    - last_sequence: the last available input sequence (scaled)
    - forecast_steps: how many steps (months) to forecast
    """
    model.eval()
    predicted = []
    current_sequence = last_sequence.clone().to(device)
    with torch.no_grad():
        for _ in range(forecast_steps):
            prediction = model(current_sequence)
            predicted.append(prediction.item())
            # Update the sequence by removing the first element and adding the prediction at the end
            current_sequence = torch.cat((current_sequence[:, 1:, :], prediction.view(1, 1, 1)), dim=1)
    predicted = np.array(predicted).reshape(-1, 1)
    predicted = scaler.inverse_transform(predicted)
    return predicted

# ---------------- Main Routine ----------------

def main():
    # File path and parameters
    file_path = 'result/40708spi.txt'
    spi_column = 'spi1'
    look_back = 12  # Use past 12 months to predict the next value

    # Load and preprocess data
    data = load_data(file_path, column=spi_column)
    spi_series = data[spi_column]
    spi_values = spi_series.values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(spi_values)

    # Create sequences for training
    X, y = create_dataset(scaled_data, look_back)
    X_tensor = torch.from_numpy(X).float().unsqueeze(2)  # Shape: (samples, look_back, 1)
    y_tensor = torch.from_numpy(y).float()

    # Create Dataset and DataLoader
    dataset = SPIDataset(X_tensor, y_tensor)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # Check if a pre-trained model checkpoint exists; if not, train the model
    checkpoint_path = 'lstm_model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded pre-trained model.")
    else:
        print("Training model...")
        train_model(model, dataloader, criterion, optimizer, num_epochs, device)
        torch.save(model.state_dict(), checkpoint_path)
        print("Model trained and saved.")

    # ---------------- One-Step Forecast ----------------
    last_sequence = scaled_data[-look_back:]
    last_sequence_tensor = torch.from_numpy(last_sequence).float().unsqueeze(0).unsqueeze(2).to(device)
    model.eval()
    with torch.no_grad():
        forecast = model(last_sequence_tensor)
    forecast_value = scaler.inverse_transform(forecast.cpu().numpy())
    print("Forecasted SPI for next month:", forecast_value[0][0])

    # ---------------- Multi-Step Forecast (Next 12 Months) ----------------
    last_sequence_iter = torch.from_numpy(scaled_data[-look_back:]).float().unsqueeze(0).to(device)
    predicted_spi = forecast_future(model, last_sequence_iter, scaler, forecast_steps=12, device=device)
    forecast_dates = pd.date_range(start=spi_series.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_SPI': predicted_spi.flatten()})
    print("12-Month Forecast for SPI:")
    print(forecast_df)

    # ---------------- Plotting ----------------
    plt.figure(figsize=(12, 6))
    plt.plot(spi_series.index, spi_series.values, label='Historical SPI')
    plt.plot(forecast_df['Date'], forecast_df['Predicted_SPI'], marker='o', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('SPI Value')
    plt.title('SPI Forecast using LSTM')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

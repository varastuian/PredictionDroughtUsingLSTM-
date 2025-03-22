import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def load_data(file_path, column='spi1', date_format='%m/%d/%Y', delimiter=' ',):

    data = pd.read_csv(file_path, delimiter=delimiter)
    data['date'] = pd.to_datetime(data['date'], format=date_format)
    data.set_index('date', inplace=True)
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

    look_back = 12  # Use past 12 months to predict the next value

    # Load and preprocess data
    data = load_data('result/40708spi.txt', column='spi1')
    spi_series = data['spi1']
    start_date = "1990-01-01"
    end_date = "1996-01-01"
    spi_series = spi_series.loc[start_date:end_date]
    spi_series.plot(title="SPI-1", figsize=(10, 5), marker='o')
    
    spi_values = spi_series.values.reshape(-1, 1)



    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(spi_values)


    # ---------------- Train-Test Split ----------------
    # Use 80% of the data for training and the remaining for testing.
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    # For the test set, include an overlap of look_back to properly form sequences.
    # test_data = scaled_data[train_size - look_back:]
    test_data = scaled_data[train_size :]

    # Create datasets for training and testing
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Convert arrays to tensors
    X_train_tensor = torch.from_numpy(X_train).float().unsqueeze(2)  # (samples, look_back, 1)
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float().unsqueeze(2)
    y_test_tensor = torch.from_numpy(y_test).float()

    # Create Dataset and DataLoader for training
    train_dataset = SPIDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # Check if a pre-trained model checkpoint exists; if not, train the model
    checkpoint_path = 'lstm_model_checkpoint2.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded pre-trained model.")
    else:
        print("Training model...")
        train_model(model, train_loader, criterion, optimizer, num_epochs, device)
        torch.save(model.state_dict(), checkpoint_path)
        print("Model trained and saved.")

# ---------------- Validate on Test Set ----------------
    # We'll perform one-step predictions for each test sequence.
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            input_seq = X_test_tensor[i].unsqueeze(0).to(device)  # shape: (1, look_back, 1)
            pred = model(input_seq)
            predictions.append(pred.item())

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))


    # ---------------- One-Step Forecast ----------------
    last_sequence = scaled_data[-look_back:]
    # last_sequence_tensor = torch.from_numpy(last_sequence).float().unsqueeze(0).unsqueeze(2).to(device)
    last_sequence_tensor = torch.from_numpy(last_sequence).float().unsqueeze(0).to(device)  # Shape: (1, look_back, 1)

    model.eval()
    with torch.no_grad():
        forecast = model(last_sequence_tensor)
    forecast_value = scaler.inverse_transform(forecast.cpu().numpy())
    print("Forecasted SPI for next month:", forecast_value[0][0])

    # ---------------- Multi-Step Forecast (Next 12 Months) ----------------
    predicted_spi = forecast_future(model, last_sequence_tensor, scaler, forecast_steps=12, device=device)
    forecast_dates = pd.date_range(start=spi_series.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_SPI': predicted_spi.flatten()})
    print("12-Month Forecast for SPI:")
    print(forecast_df)

    # ---------------- Plotting ----------------

    test_dates = spi_series.index[train_size:][:len(y_test)]

    plt.figure(figsize=(12, 6))
    plt.plot(spi_series.index, spi_series.values, label='Historical SPI')
    plt.plot(forecast_df['Date'], forecast_df['Predicted_SPI'], marker='o', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('SPI Value')
    plt.title('SPI Forecast using LSTM')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

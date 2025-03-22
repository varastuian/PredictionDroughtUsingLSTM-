import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os

# Load dataset
df = pd.read_csv('result/40708spi.txt', delimiter=' ')
timeseries = df[['spi1']].values.astype('float32')

# Train-test split for time series
train_size = int(len(timeseries) * 0.67)
train, test = timeseries[:train_size], timeseries[train_size:]

# Function to create dataset with improved conversion
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback])
    # Convert lists to np.array then to tensor for better performance
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lookback = 24
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        # x shape: (batch, sequence, feature)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])  # Take output from last time step
        return x

# Initialize model
model = LSTMModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

checkpoint_path = 'lstm_model_checkpoint2.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded pre-trained model.")
else:
    print("Training model...")
    n_epochs = 800
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation every 100 epochs
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_rmse = np.sqrt(loss_fn(model(X_train), y_train).item())
                test_rmse = np.sqrt(loss_fn(model(X_test), y_test).item())
            print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")
    torch.save(model.state_dict(), checkpoint_path)
    print("Model trained and saved.")

# Get predictions for train and test sets
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train).detach().cpu().numpy().flatten()
    y_test_pred = model(X_test).detach().cpu().numpy().flatten()

# Create 1D arrays for plotting actual vs predicted values
train_plot = np.full((len(timeseries),), np.nan, dtype=np.float32)
# Train predictions will start from index 'lookback' up to train_size
train_plot[lookback:train_size] = y_train_pred

test_plot = np.full((len(timeseries),), np.nan, dtype=np.float32)
# Test predictions will start from index 'train_size + lookback'
test_plot[train_size+lookback:] = y_test_pred

# Plot actual vs predicted values for train and test
plt.figure(figsize=(12, 6))
plt.plot(timeseries.flatten(), label="Actual SPI", color='blue')
plt.plot(train_plot, label="Train Predictions", color='red')
plt.plot(test_plot, label="Test Predictions", color='green')
plt.xlabel("Time")
plt.ylabel("SPI")
plt.title("SPI Forecasting using LSTM")
plt.legend()
plt.show()

# Generate 24-month future forecast
future_input = torch.tensor(test[-lookback:], dtype=torch.float32).unsqueeze(0)  # shape: (1, lookback, 1)
future_preds = []

for _ in range(24):
    with torch.no_grad():
        pred = model(future_input).item()
    future_preds.append(pred)
    # Update future_input by appending the new prediction and removing the oldest entry
    future_input = torch.cat((future_input[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)), dim=1)

# Create forecast dataframe
forecast_dates = pd.date_range(start=pd.Timestamp.today(), periods=24, freq='MS')  # Monthly frequency
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_SPI': future_preds})

# Print forecast values
print("\n24-Month Forecast for SPI:")
print(forecast_df)

# Plot future forecast
plt.figure(figsize=(10, 4))
plt.plot(forecast_df['Date'], forecast_df['Predicted_SPI'], marker='o', label='Forecast')
plt.xlabel("Date")
plt.ylabel("SPI")
plt.title("SPI Forecast for Next 24 Months")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('./result/40708spi.txt', sep=' ')
data.set_index('date', inplace=True)

data.replace(-99, np.nan, inplace=True)
spi_series = data['spi1'].dropna()
spi_values = spi_series.values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(spi_values)

# Create dataset function
def create_dataset(dataset, look_back=12):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, 0])
        y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(y)

look_back = 12  # Using past 12 months to predict the next value
X, y = create_dataset(scaled_data, look_back)
X_tensor = torch.from_numpy(X).float().unsqueeze(2)
y_tensor = torch.from_numpy(y).float()

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_data, test_data = random_split(list(zip(X_tensor, y_tensor)), [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Initialize model, loss function, and optimizer
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2600
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'lstm_spi_forecast.pth')

# Evaluate on test data
model.eval()
test_losses = []
actuals, predictions = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        test_losses.append(loss.item())
        
        # Save actual and predicted values
        actuals.append(targets.item())
        predictions.append(outputs.item())

print(f'Average Test Loss: {np.mean(test_losses):.4f}')

# Convert predictions back to original scale
actuals = np.array(actuals).reshape(-1, 1)
predictions = np.array(predictions).reshape(-1, 1)
actuals = scaler.inverse_transform(actuals)
predictions = scaler.inverse_transform(predictions)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Actual SPI')
plt.plot(predictions, label='Predicted SPI', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('SPI Value')
plt.title('SPI Forecast using LSTM')
plt.legend()
plt.show()

# Forecast next value
# last_sequence = torch.from_numpy(scaled_data[-look_back:]).float().unsqueeze(0).unsqueeze(2).to(device)
last_sequence = scaled_data[-look_back:]  # Extract last `look_back` values
last_sequence = torch.from_numpy(last_sequence).float().unsqueeze(0).to(device)  # Shape: (1, look_back)
last_sequence = last_sequence.unsqueeze(2)  # Shape: (1, look_back, 1) -> (batch_size, sequence_length, input_size)

with torch.no_grad():
    forecast = model(last_sequence)
forecast_value = scaler.inverse_transform(forecast.cpu().numpy())
print("Forecasted SPI for next month:", forecast_value[0][0])

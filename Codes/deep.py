import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Codes\spi_results.csv')
df['date'] = pd.to_datetime(df['data'])
station_df = df[df['station_id'] == 40708]
january_df = station_df[station_df['date'].dt.month == 1]

# Display the result
# print(january_df)
# print(station_df.head())

# # View the last 5 rows
# print(station_df.tail())

# # View the entire DataFrame
# plt.figure(figsize=(12, 6))
# plt.plot(station_df['date'], station_df['SPI'], label='SPI', color='blue')
# plt.title('SPI Time Series')
# plt.xlabel('Date')
# plt.ylabel('SPI')
# plt.legend()
# plt.grid(True)
# plt.show()

# features = ['tmax_m', 'tmax_max', 'tmax_min', 'tmin_m', 'tmin_min', 'tmin_max', 'ntmin_0', 'rrr24', 'sshn', 'tm_m', 't_03_m', 't_09_m', 't_15_m']
features = ['tmax_m', 'tmin_m', 'rrr24', 'SPI']

target = 'SPI'

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features].values)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        target = data[i+window_size, -1]  # target is the last column (SPI)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)




# Hyperparameters
input_size = len(features)
hidden_size = 64
num_layers = 2
output_size = 1
# seq_length = 10
window_size = 12  # 12 months = 1 year

seq_length = window_size

batch_size = 32
num_epochs = 100
learning_rate = 0.001


X, y = create_sequences(scaled_data, window_size)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
print(f"Using {device} device")



dataset = TimeSeriesDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
model.train()

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'lstm_model_vahid.pth')
# Load the saved model for inference
model.load_state_dict(torch.load('lstm_model_vahid.pth'))
model.eval()
# Prepare test data for prediction
# test_data = torch.tensor(scaled_data[-seq_length:], dtype=torch.float32).unsqueeze(0)
# test_data = scaled_data[-window_size:]  # Ensure 12 months
# test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(device)

# # Make predictions
# with torch.no_grad():
#     predictions = model(test_data)
# predictions_np = predictions.cpu().numpy()
# dummy_array = np.zeros((predictions_np.shape[0], scaler.min_.shape[0]))
# dummy_array[:, -1] = predictions_np[:, 0] 
# # Inverse transform predictions to original scale
# # predictions_rescaled = scaler.inverse_transform(predictions.numpy())
# predictions_rescaled = scaler.inverse_transform(dummy_array)
# predicted_spi = predictions_rescaled[:, -1]

# # Print the predictions
# print("Predicted SPI:", predicted_spi)

last_sequence = scaled_data[-window_size:]  # Assuming scaled_data contains data up to 2024
last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

# Predict SPI for 2025
predicted_spi = []
for i in range(12):  # Predict for 12 months
    with torch.no_grad():
        prediction = model(last_sequence)
        predicted_spi.append(prediction.item())

        # Update the sequence with the predicted value
        next_input = torch.cat((last_sequence[:, 1:, :], prediction.unsqueeze(0).unsqueeze(2)), dim=1)
        last_sequence = next_input

# Convert predictions to a DataFrame for visualization
predicted_spi_df = pd.DataFrame({
    'Month': [f'2025-{i+1:02d}' for i in range(12)],
    'Predicted_SPI': predicted_spi
})

# Print the predictions
print(predicted_spi_df)

# Plot the predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(predicted_spi_df['Month'], predicted_spi_df['Predicted_SPI'], marker='o', label='Predicted SPI')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('SPI')
plt.title('Predicted SPI for 2025')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

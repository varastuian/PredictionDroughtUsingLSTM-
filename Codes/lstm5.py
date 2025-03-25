import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('result/40708spi.txt', delim_whitespace=True)
# Parse the date column (format = mm/dd/yyyy )
# df['date'] = pd.to_datetime(df['date'], dayfirst=False)

timeseries = df['spi1'].values.astype('float32')
# dates = df['date'].values  
# scaler = MinMaxScaler(feature_range=(0, 1))
# timeseries = scaler.fit_transform(timeseries.reshape(-1, 1)).flatten()

train_size = int(len(timeseries) * 0.67)
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i + lookback])
        # y.append(dataset[i + lookback])
        y.append(dataset[i+1:i+lookback+1])

    X = np.array(X)
    y = np.array(y)
    # Reshape X to include the feature dimension (needed by LSTM: batch, seq, feature)
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
lookback = 24

X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)
# Move data to GPU
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)



class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2)

        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        # x: (batch, sequence, feature)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

model = LSTMModel().to(device)

# optimizer = optim.Adam(model.parameters(), lr=0.005)
optimizer = optim.RMSprop(model.parameters(), lr=0.005)

loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=16)

checkpoint_path = 'lstm_model_checkpoint55.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded pre-trained model.")
else:
    train_losses, test_losses = [], []

    for epoch in range(800):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move batch data to GPU

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_rmse = np.sqrt(loss_fn(model(X_train), y_train).item())
                test_rmse = np.sqrt(loss_fn(model(X_test), y_test).item())
                train_losses.append(train_rmse)
                test_losses.append(test_rmse)
            print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")
    torch.save(model.state_dict(), checkpoint_path)
    print("Model trained and saved.")
    plt.plot(train_losses, label='Train RMSE')
    plt.plot(test_losses, label='Test RMSE')
    plt.xlabel("Epochs (x10)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()
# --- Get Predictions ---
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train).detach().cpu().numpy().flatten()
    y_test_pred = model(X_test).detach().cpu().numpy().flatten()
# y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

train_plot = np.full((len(timeseries),), np.nan, dtype=np.float32)
train_plot[lookback:train_size] = y_train_pred

test_plot = np.full((len(timeseries),), np.nan, dtype=np.float32)
test_plot[train_size+lookback:] = y_test_pred


test_start_idx = train_size + lookback

test_actual = timeseries[test_start_idx:]
# test_dates = dates[test_start_idx:]
# test_actual = scaler.inverse_transform(test_actual.reshape(-1, 1)).flatten()

plt.figure(figsize=(12,6))
plt.plot(test_actual, label="Actual SPI (Test)", marker='o')
plt.plot(y_test_pred, label="Predicted SPI (Test)", marker='x', linestyle='--')
plt.xlabel("Date")
plt.ylabel("SPI")
plt.title("Comparison of Actual vs. Predicted SPI (Test Data)")
plt.legend()
plt.grid(True)
plt.show()


forecast_df = pd.DataFrame({'Date': test_dates, 'actual':test_actual,'Predicted_SPI': y_test_pred})

print(forecast_df) 

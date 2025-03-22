import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os

df = pd.read_csv('result/40708spi.txt', delim_whitespace=True)
# Parse the date column (format = mm/dd/yyyy )
df['date'] = pd.to_datetime(df['date'], dayfirst=False)

timeseries = df['spi1'].values.astype('float32')
dates = df['date'].values  

train_size = int(len(timeseries) * 0.67)
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback])
    X = np.array(X)
    y = np.array(y)
    # Reshape X to include the feature dimension (needed by LSTM: batch, seq, feature)
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

lookback = 24
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        # x: (batch, sequence, feature)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

model = LSTMModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

checkpoint_path = 'lstm_model_checkpoint5.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded pre-trained model.")
else:
    for epoch in range(800):
        model.train()
        for X_batch, y_batch in loader:
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
            print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")
    torch.save(model.state_dict(), checkpoint_path)
    print("Model trained and saved.")

# --- Get Predictions ---
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train).detach().cpu().numpy().flatten()
    y_test_pred = model(X_test).detach().cpu().numpy().flatten()

train_plot = np.full((len(timeseries),), np.nan, dtype=np.float32)
train_plot[lookback:train_size] = y_train_pred

test_plot = np.full((len(timeseries),), np.nan, dtype=np.float32)
test_plot[train_size:] = y_test_pred


test_start_idx = train_size + lookback

test_actual = timeseries[test_start_idx:]
test_dates = dates[test_start_idx:]

plt.figure(figsize=(12,6))
plt.plot(test_dates, test_actual, label="Actual SPI (Test)", marker='o')
plt.plot(test_dates, y_test_pred, label="Predicted SPI (Test)", marker='x', linestyle='--')
plt.xlabel("Date")
plt.ylabel("SPI")
plt.title("Comparison of Actual vs. Predicted SPI (Test Data)")
plt.legend()
plt.grid(True)
plt.show()


forecast_df = pd.DataFrame({'Date': test_dates, 'actual':test_actual,'Predicted_SPI': y_test_pred})

print(forecast_df) 

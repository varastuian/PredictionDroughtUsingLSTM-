import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os

df = pd.read_csv('result/40708spi.txt', delimiter=' ')
# df.info()
df.head()
df.shape
timeseries = df[['spi1']].values.astype('float32')
df[['spi1']].plot()
plt.show()
# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 24
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

checkpoint_path = 'lstm_model_checkpoint.pth'
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
            # print("Epoch %d" % epoch)

        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    torch.save(model.state_dict(), checkpoint_path)
    print("Model trained and saved.")
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]


last_sequence = timeseries[-lookback:]
model.eval()
predicted = []
current_sequence = last_sequence.clone()
with torch.no_grad():
    for _ in range(lookback):
        prediction = model(current_sequence)
        predicted.append(prediction.item())
        # Update the sequence by removing the first element and adding the prediction at the end
        current_sequence = torch.cat((current_sequence[:, 1:, :], prediction.view(1, 1, 1)), dim=1)
predicted = np.array(predicted).reshape(-1, 1)

forecast_dates = pd.date_range(start=timeseries.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_SPI': predicted.flatten()})
print("12-Month Forecast for SPI:")
print(forecast_df)
# plot
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.plot(forecast_df['Date'], forecast_df['Predicted_SPI'], marker='o', label='Forecast')

plt.show()
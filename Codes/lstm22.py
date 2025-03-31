import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import mean_squared_error
import pywt

df = pd.read_csv('result/40708spi.txt', delimiter=' ')
# timeseries = df[['spi1']].values.astype('float32')
timeseries = df['spi1'].values.astype('float32')

wavelet = 'db4'
level = 2
coeffs = pywt.wavedec(timeseries, wavelet, level=level)
# Approximation: a denoised/smoothed version
approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(timeseries))
# Detail from level 1 (used for boosted methods)
detail = pywt.upcoef('d', coeffs[1], wavelet, level=level, take=len(timeseries))
timeseries_org = timeseries
timeseries = approx
# train-test split for time series
train_size = int(len(timeseries) * 0.67)
# test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)
class lstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = lstmModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()


# loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)


train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 8
loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

checkpoint_path = 'lstm_model_checkpoint22.pth'
# if os.path.exists(checkpoint_path):
if False:
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded pre-trained model.")
else:
    print("Training model...")

    n_epochs = 300
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 50 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))

        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    # torch.save(model.state_dict(), checkpoint_path)
    # print("Model trained and saved.")


with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# plot
plt.figure(figsize=(14,12))

plt.plot(timeseries_org,label='actual spi',marker='o')
plt.plot(train_plot,label='trained spi', c='r',marker='o')
plt.plot(test_plot, label='test spi',c='g',marker='o')
plt.legend()

plt.show()



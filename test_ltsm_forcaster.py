import pandas as pd
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster

# df = pd.read_csv('AirPassengers.csv',parse_dates=['Month'])

df = pd.read_csv('result/40708spi.txt', delimiter=' ')
df['date'] = pd.to_datetime(df['date'],dayfirst=False)

f = Forecaster(y=df['spi1'],current_dates=df['date'])

# print(f)

f.plot_pacf(lags=26)
plt.show()


f.seasonal_decompose().plot()
plt.show()


stat, pval, _, _, _, _ = f.adf_test(full_res=True)

print(stat,pval)



# Set the test length (how many data points to leave out for validation)
f.set_test_length(12)

# Generate future dates for forecasting
f.generate_future_dates(12) # Forecast for 12 more months


# Add multiple models
f.set_estimator('prophet')
f.manual_forecast()

f.set_estimator('arima')
f.manual_forecast(order=(2,1,2))  # or use auto_arima=True for tuning

f.set_estimator('lightgbm')
f.manual_forecast(
    lags=12,  # use 12 lagged values
    alpha=0.05,  # confidence interval
    verbose=False,
)

# Set the LSTM estimator
f.set_estimator('lstm')

f.manual_forecast(
    lags=36,
    batch_size=32,
    epochs=15,
    validation_split=.2,
    activation='tanh',
    optimizer='Adam',
    learning_rate=0.001,
    lstm_layer_sizes=(100,)*3,
    dropout=(0,)*3,
)

f.tune()
# Predict the future values
# f.predict()
f.auto_forecast(call_me='auto')
# Plot the actual vs predicted values
# f.plot_test_set()
f.plot()
# Plot the future forecast
# f.plot_forecast()

# Show the plots
plt.show()
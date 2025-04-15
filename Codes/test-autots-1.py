import pandas as pd
import numpy as np
from autots import AutoTS

# Load your data
df = pd.read_csv('result/40708spi.txt', delimiter=' ')

# Convert the 'date' column to datetime.
# Adjust the format if your date is in day/month/year format.
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

# Check if any dates failed to parse
if df['date'].isnull().any():
    raise ValueError("Some dates couldn't be parsed. Please check your date format.")

# Replace -99 with NaN for proper missing value handling.
df.replace(-99, np.nan, inplace=True)

# Optional: sort data by date
df = df.sort_values('date')

# Fill missing values in 'spi1'; here using forward-fill as an example.
df['spi1'] = df['spi1'].fillna(method='ffill')
if df['spi1'].isnull().any():
    raise ValueError("After filling, 'spi1' still contains missing values.")

# Select which columns you want to forecast; here we forecast 'spi1'
value_cols = ['spi1']

# Create and configure the AutoTS model
model = AutoTS(
    forecast_length=12,      # Forecast 12 time periods ahead
    frequency='infer',       # Let AutoTS infer the frequency from your data
    ensemble='simple'        # Use a simple ensemble of top models
)

# Fit the model. Specify the date column and value columns.
model = model.fit(
    df, 
    date_col='date',
    value_cols=value_cols
)

# Make predictions
prediction = model.predict()

# Extract and print the forecast
forecast = prediction.forecast
print(forecast)

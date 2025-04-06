import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from scalecast.Pipeline import Transformer, Reverter, Pipeline
from scalecast.util import (
    find_optimal_transformation,
    gen_rnn_grid,
    backtest_for_resid_matrix,
    get_backtest_resid_matrix,
    overwrite_forecast_intervals,
    infer_apply_Xvar_selection,
)
from scalecast import GridGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# import pandas_datareader as pdr

# data = pd.read_csv('AirPassengers.csv',parse_dates=['Month'])
data = pd.read_csv('result/40708spi.txt', delimiter=' ')
data['date'] = pd.to_datetime(data['date'],dayfirst=False)  # Assuming there's a 'date' column
# data = df['spi1'].values.astype('float')
f = Forecaster(
    y=data['spi1'],
    current_dates=data['date'],
    future_dates = 24,
)


f.plot()
plt.show()

def forecaster(f):
    f.set_estimator('rnn')
    f.manual_forecast(
        lags = 18,
        layers_struct = [
            ('LSTM',{'units':36,'activation':'tanh'}),
        ],
        epochs=200,
        call_me = 'lstm',
    )

transformer = Transformer(
    transformers = [
        ('DetrendTransform',{'poly_order':2}),
        'DeseasonTransform',
    ],
)

reverter = Reverter(
    reverters = [
        'DeseasonRevert',
        'DetrendRevert',
    ],
    base_transformer = transformer,
)

pipeline = Pipeline(
    steps = [
        ('Transform',transformer),
        ('Forecast',forecaster),
        ('Revert',reverter),
    ]
)

f = pipeline.fit_predict(f)


f.plot()
plt.savefig('LSTM Univariate.png')
plt.show()
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from statsmodels.tsa.stattools import acf, pacf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

######################################
# 1. Read and Preprocess Data
######################################
data_path = 'result/40708spi.txt'
df = pd.read_csv(data_path, delimiter=' ')
series = df['spi1'].values.astype('float32')

# plt.figure(figsize=(10,4))
# plt.plot(series, marker='o', markersize=3)
# plt.title('Raw SPI Time Series')
# plt.xlabel('Time index')
# plt.ylabel('SPI')
# plt.show()
selected_lags = 12
######################################
# 4. Create Dataset
######################################
def create_dataset(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append([series[i - lags]])
        y.append(series[i])
    return np.array(X), np.array(y)

# Dataset for "approximation-based" methods:
X_all, y_all = create_dataset(series, selected_lags)
train_size = int(len(X_all) * 0.67)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]


######################################
# 5. Define Evaluation Metrics
######################################
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def correlation_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

######################################
# 7. Model 2: SVR (Using approx dataset)
######################################
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_rmse = rmse(y_test, svr_pred)
svr_nse = nse(y_test, svr_pred)
svr_cc = correlation_coef(y_test, svr_pred)
print("\nSVR Performance:")
print(f"RMSE: {svr_rmse:.4f}, NSE: {svr_nse:.4f}, CC: {svr_cc:.4f}")

######################################
# 9. Model 4: EDT (ExtraTrees) (Using approx dataset)
######################################
edt_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
edt_model.fit(X_train, y_train)
edt_pred = edt_model.predict(X_test)
edt_rmse = rmse(y_test, edt_pred)
edt_nse = nse(y_test, edt_pred)
edt_cc = correlation_coef(y_test, edt_pred)
print("\nEDT (ExtraTrees) Performance:")
print(f"RMSE: {edt_rmse:.4f}, NSE: {edt_nse:.4f}, CC: {edt_cc:.4f}")

######################################
# 10. Model 5: Random Forest (Using approx dataset)
######################################
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = rmse(y_test, rf_pred)
rf_nse = nse(y_test, rf_pred)
rf_cc = correlation_coef(y_test, rf_pred)
print("\nRandom Forest Performance:")
print(f"RMSE: {rf_rmse:.4f}, NSE: {rf_nse:.4f}, CC: {rf_cc:.4f}")

######################################
# 15. Compare All Models
######################################
results = {
    'SVR': {'RMSE': svr_rmse, 'NSE': svr_nse, 'CC': svr_cc},
    'EDT': {'RMSE': edt_rmse, 'NSE': edt_nse, 'CC': edt_cc},
    'RF': {'RMSE': rf_rmse, 'NSE': rf_nse, 'CC': rf_cc},
}
print("\n=== Model Comparison on Test Data ===")
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE = {metrics['RMSE']:.4f}, NSE = {metrics['NSE']:.4f}, CC = {metrics['CC']:.4f}")

######################################
# 16. Plot Test Predictions for All Methods (approx. 1-year test period)
######################################
plt.figure(figsize=(14,12))
plot_order = ['SVR','EDT','RF']
for i, model_name in enumerate(plot_order):

    if model_name=='SVR':
        pred = svr_pred

    elif model_name=='EDT':
        pred = edt_pred
    elif model_name=='RF':
        pred = rf_pred

    true_vals = y_test
    plt.subplot(5,2,i+1)
    plt.plot(true_vals, label='actual sppi', marker='o', markersize=3)
    plt.plot(pred, label=model_name, marker='x', linestyle='--')
    plt.title(model_name, fontsize=10)
    # plt.xlabel('Test Index')
    plt.ylabel('SPI')
    plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

######################################
# 17. Taylor Diagram for Model Comparison
######################################
# Use the approx-based predictions for models not wavelet-boosted.
obs_std = np.std(y_test)
model_names_td = ['SVR','EDT','RF']
std_devs = [ np.std(svr_pred), np.std(edt_pred), np.std(rf_pred),]
corrs = [ svr_cc, edt_cc, rf_cc]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, polar=True)
theta = [np.arccos(c) for c in corrs]
r = std_devs
for i, name in enumerate(model_names_td):
    ax.plot(theta[i], r[i], 'o', label=name)
# Plot observation as reference
ax.plot(0, obs_std, 'r*', markersize=12, label='Obs')
ax.set_title('Taylor Diagram', fontsize=14)
ax.set_rlim(0, max(std_devs + [obs_std])*1.1)
corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
theta_ticks = [np.arccos(c) for c in corr_ticks]
ax.set_thetagrids(np.degrees(theta_ticks), labels=[str(c) for c in corr_ticks])
# Place legend outside the plot area
plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=8)
plt.show()
######################################
# 18. Forecast 10 Years (120 Months) into the Future using the Best Model
######################################
# Select best model based on RMSE among all nine methods.
model_rmse_dict = {name: metrics['RMSE'] for name, metrics in results.items()}
best_model_name = min(model_rmse_dict, key=model_rmse_dict.get)
print(f"\nBest model selected for forecasting: {best_model_name}")

n_forecast = 320
forecast = []


# For non-wavelet-boosted methods, use processed_series and its lags.
base_series = series
current_input = base_series[-max(12):].tolist()  # length = max(selected_lags)
for i in range(n_forecast):
    X_input = np.array(current_input).astype('float32')
    if best_model_name == 'SVR':
        next_val = svr_model.predict(X_input.reshape(1, -1))[0]

    elif best_model_name == 'EDT':
        next_val = edt_model.predict(X_input.reshape(1, -1))[0]
    elif best_model_name == 'RF':
        next_val = rf_model.predict(X_input.reshape(1, -1))[0]
    forecast.append(next_val)
    current_input.pop(0)
    current_input.append(next_val)

# Plot the forecast along with the historical series.
plt.figure(figsize=(10,5))
base_series = series
plt.plot(range(len(base_series)), base_series, label='Historical Series')
plt.plot(range(len(base_series), len(base_series)+n_forecast), forecast, label='10-Year Forecast', color='red', marker='o', linestyle='--')
plt.title(f'10-Year Forecast using {best_model_name}')
plt.xlabel('Time Index (months)')
plt.ylabel('SPI')
plt.legend()
plt.show()

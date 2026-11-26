import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import time

data = pd.read_csv("Climate Data.csv") #File Directory Here
data["date"] = pd.to_datetime(data["date"], format='%d/%m/%Y')
data.set_index(data['date'], inplace=True)
# Variables: Heat Index, UV Radiation, Wind Speed, Solar Radiation
data = data[['date', 'adm4_pcode', 'heat_index', 'solar_rad', 'wind_speed', 'uv_rad']]
clean_data = data[["heat_index", "solar_rad", "wind_speed", "uv_rad"]].resample('W').mean().dropna()
heat_index_weekly = clean_data["heat_index"]

# Split Training Data
heat_index_weekly_train = heat_index_weekly.loc[:'2019-12-29']
heat_index_weekly_test = heat_index_weekly.loc['2020-01-05':]

# SARIMAX
exo_raw = clean_data[["solar_rad", "wind_speed", "uv_rad"]]
exo_features = pd.concat([exo_raw.shift(52), exo_raw.shift(1)], axis=1).dropna()
exo_features_train = exo_features.loc[:"2019-12-29"]
exo_features_test = exo_features.loc["2020-01-05":]
sarimax_train = heat_index_weekly.loc[exo_features_train.index]
exo_features_train = exo_features_train.loc[sarimax_train.index]

# D = 1, d = 0 (Stationary)
heat_D = (heat_index_weekly - heat_index_weekly.shift(52)).dropna()
# heat_D = heat_D.loc[:"2019-12-29"]

# Checks
def ADF(heat):
    
    adf = adfuller(heat)
    KPSS = kpss(heat, regression='c')
    a_p = adf[1]

    if a_p < 0.05:
        print(f'Augmented Dicker Fuller: \n p-value: {a_p} \n Sample Size: {adf[3]}')
        print("Passed ADF")
    k_p = KPSS[1]
    if k_p > 0.05:
        print(f'KPSS Test: \n p-value: {k_p}')
        print("Passed KPSS")
    if a_p < 0.05 and k_p > 0.05:
        print("The data is Stationary!")
        plt.plot(heat_D.index, heat_D, color='#800000')
        plt.ylabel('Heat Index (D = 1)')
        plt.xlabel('Year')
        plt.title('Time Series of Heat Index')
        plt.show()

def ARIMA(heat):
    arima = pm.auto_arima(
        heat,
        start_p=0,
        max_p=2,
        start_q=0,
        max_q=2,
        seasonal=False,
        stepwise=True,
        trace=True,
        error_action='ignore'
    )
    return arima.summary()

def SARIMA(sheat):
    sarima = pm.auto_arima(
        sheat,
        seasonal=True,
        start_p=0,
        max_p=2,
        start_q=0,
        max_q=2,
        start_P=0,
        max_P=1,
        m=52,
        start_Q=0,
        D=1,
        max_Q=1,
        stepwise=True,
        trace=True,
        error_action='ignore'
    )
    return sarima.summary()

# def SARIMAX(sxheat, x):
#     sarimax = pm.auto_arima(
#         sxheat,
#         seasonal=True,
#         start_p=0,
#         exog=x,
#         max_p=2,
#         start_q=0,
#         max_q=2,
#         start_P=0,
#         max_P=1,
#         m=52,
#         start_Q=0,
#         D=1,
#         max_Q=1,
#         stepwise=True,
#         trace=True,
#         error_action='ignore'
#     )
#     return sarimax.summary()

# Forecasting: ARIMA(1,0,1), SARIMA(1,0,2)(1,1,0)[52], SARIMAX(2,0,1)(1,1,0)[52]
def SARIMA_Walk_Forward(y_train, y_test):
  start = time.perf_counter()
  sarima = SARIMAX(
      y_train,
      order=(1, 0, 2),
      seasonal_order=(1, 1, 0, 52),
      enforce_stationarity=False,
      enforce_invertibility=False,
  )
  smodel_fit = sarima.fit(disp=False)

  predictions = []
  for t in range(len(y_test)):
    walk = smodel_fit.forecast(steps=1).iloc[0]
    predictions.append(walk)
    smodel_fit = smodel_fit.append(y_test.iloc[t : t + 1], refit=False)

  run_time = time.perf_counter() - start

  y_true = np.asarray(y_test)
  y_pred = np.asarray(predictions)
  
  mae = mean_absolute_error(y_true, y_pred)
  bias = np.mean(y_pred - y_true)  # forecast - true
  maxerror = max_error(y_true, y_pred)
  r_squared = r2_score(y_true, y_pred)
  print(f'MAE for SARIMA Walk Forward: {mae}')  
  print(f'Bias for SARIMA Walk Forward: {bias}')
  print(f'Max Error for SARIMA Walk Forward: {maxerror}')
  print(f'R Squared for SARIMA Walk Forward: {r_squared}')
  print(f'Run time for SARIMA Walk Forward: {run_time}')

  plt.plot(heat_index_weekly_test.index, heat_index_weekly_test, label='Actual (True)', color='black', linewidth=1.5, alpha=0.8)
  plt.plot(heat_index_weekly_test.index, predictions, label='SARIMAX Predicted', color='#d9534f', linestyle='--', linewidth=2)
  plt.title('SARIMA Walk Forward Prediction')
  plt.xlabel('Date') 
  plt.xticks(rotation=45)
  plt.ylabel('Heat Index (°C)')
  plt.show()
  return
def ARIMA_Walk_Forward(y_train, y_test):
  start = time.perf_counter()
  sarima = SARIMAX(
      y_train,
      order=(1, 0, 1),
      seasonal_order=(0, 0, 0, 0),
      enforce_stationarity=False,
      enforce_invertibility=False,
  )
  smodel_fit = sarima.fit(disp=False)

  predictions = []
  for t in range(len(y_test)):
    walk = smodel_fit.forecast(steps=1).iloc[0]
    predictions.append(walk)
    smodel_fit = smodel_fit.append(y_test.iloc[t : t + 1], refit=False)

  run_time = time.perf_counter() - start

  y_true = np.asarray(y_test)
  y_pred = np.asarray(predictions)
  
  mae = mean_absolute_error(y_true, y_pred)
  bias = np.mean(y_pred - y_true)  # forecast - true
  maxerror = max_error(y_true, y_pred)
  r_squared = r2_score(y_true, y_pred)
  print(f'MAE for ARIMA Walk Forward: {mae}')  
  print(f'Bias for ARIMA Walk Forward: {bias}')
  print(f'Max Error for ARIMA Walk Forward: {maxerror}')
  print(f'R Squared for ARIMA Walk Forward: {r_squared}')
  print(f'Run time for ARIMA Walk Forward: {run_time}')

  plt.plot(heat_index_weekly_test.index, heat_index_weekly_test, label='Actual (True)', color='black', linewidth=1.5, alpha=0.8)
  plt.plot(heat_index_weekly_test.index, predictions, label='SARIMAX Predicted', color='#d9534f', linestyle='--', linewidth=2)
  plt.title('ARIMA Walk Forward Prediction')
  plt.xlabel('Date') 
  plt.xticks(rotation=45)
  plt.ylabel('Heat Index (°C)')
  plt.show()
  return

def SARIMAX_Walk_Forward(y_train, x_train, y_test, x_test):
    start = time.perf_counter()
    sarimax = SARIMAX(
        y_train,
        exog=x_train,
        order=(2, 0, 1),
        seasonal_order=(1, 1, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    xmodel_fit = sarimax.fit(disp=False)
    predictions = []
    for t in range(len(y_test)):
        # One-step-ahead forecast using the known exog for this step
        walk = xmodel_fit.forecast(steps=1, exog=x_test.iloc[t : t + 1].values).iloc[0]
        predictions.append(walk)
        # Reveal the true value (and its exog) to the model, no refit
        xmodel_fit = xmodel_fit.append(
            y_test.iloc[t : t + 1],
            exog=x_test.iloc[t : t + 1],
            refit=False,
        )
    run_time = time.perf_counter() - start
    y_true = np.asarray(y_test)
    y_pred = np.asarray(predictions)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)  # forecast - true
    maxerror = max_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    print(f'MAE for SARIMAX Walk Forward (2,0,1)(1,1,0)[52]: {mae}')
    print(f'Bias for SARIMAX Walk Forward (2,0,1)(1,1,0)[52]: {bias}')
    print(f'RMSE for SARIMAX Walk Forward (2,0,1)(1,1,0)[52]: {rmse}')
    print(f'Max Error for SARIMAX Walk Forward (2,0,1)(1,1,0)[52]: {maxerror}')
    print(f'R Squared for SARIMAX Walk Forward (2,0,1)(1,1,0)[52]: {r_squared}')
    print(f'Run time for SARIMAX Walk Forward (2,0,1)(1,1,0)[52]: {run_time}')
    print('------------------------------------------')
    plt.plot(heat_index_weekly_test.index, heat_index_weekly_test, label='Actual (True)', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(heat_index_weekly_test.index, predictions, label='SARIMAX Predicted', color='#d9534f', linestyle='--', linewidth=2)
    plt.title('SARIMAX Walk Forward Prediction')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Heat Index (°C)')
    plt.show()
    return


# # Functions to Call
# ADF(heat_D)
# ARIMA(heat_index_weekly_train)
# SARIMA(heat_index_weekly_train)
# SARIMAX(sarimax_train, exo_features_train.to_numpy())
# SARIMA_Walk_Forward(heat_index_weekly_train, heat_index_weekly_test)
# ARIMA_Walk_Forward(heat_index_weekly_train, heat_index_weekly_test)
# SARIMAX_Walk_Forward(sarimax_train, exo_features_train, heat_index_weekly_test, exo_features_test)

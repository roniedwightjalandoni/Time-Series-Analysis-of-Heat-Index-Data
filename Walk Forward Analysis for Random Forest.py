from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import time 
import pandas as pd


data = pd.read_csv("C:\\Users\\Yani\\Downloads\\TSA\\Climate Data.csv")
data["date"] = pd.to_datetime(data["date"],format='%d/%m/%Y')
data.set_index(data['date'], inplace = True)
data = data[['date','heat_index','solar_rad','wind_speed','uv_rad']]
clean_data = data[["heat_index", "solar_rad", "wind_speed", "uv_rad"]].resample('W').mean().dropna()
heat_index_weekly = clean_data["heat_index"]
exo_features = clean_data[["solar_rad", "wind_speed", "uv_rad"]]

heat_index_weekly_adjusted= heat_index_weekly.shift(52).dropna()
# print(heat_index_weekly_adjusted)
heat_D = (heat_index_weekly-heat_index_weekly.shift(52)).dropna()
new_data = pd.concat([heat_D,exo_features.shift(52), exo_features.shift(1), heat_D.shift(1).dropna(), heat_D.shift(52).dropna()],axis = 1).dropna()

x = new_data.iloc[:, 1:]
y = new_data.iloc[:, 0]

new_data_train = new_data.loc[:"2019-12-29"]
new_data_test = new_data.loc["2020-01-05":]
x_test = x.loc["2020-01-05":]
y_test = y.loc["2020-01-05":]

start = time.perf_counter()
forecast = []
test_dates = y.index[len(new_data_train) :]
for t in range(len(new_data_train), len(new_data)):
    x_train = x.iloc[:t]
    y_train = y.iloc[:t]
    model = RandomForestRegressor(
        n_estimators = 150,
        max_depth = 5,
        random_state = 42,
        n_jobs = -1
        )
    model.fit(x_train,y_train)
    Delta_prediction = model.predict(x.iloc[[t]])[0]
    prediction = heat_index_weekly_adjusted.loc[x.index[t]] + Delta_prediction
    forecast.append(prediction)
end = time.perf_counter()

plt.plot(test_dates,heat_index_weekly.loc["2020-01-05":],color='black',linewidth = 1.5)
plt.plot(test_dates,forecast,color = '#d9534f',label='Random Forest Forecast',linestyle = '--',linewidth=2.0)
plt.title('Random Forest Walk Forward Prediction')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Heat Index  (°C)')
plt.show()
rmse = np.sqrt(mean_squared_error(forecast,heat_index_weekly.loc["2020-01-05":]))
mae = mean_absolute_error(forecast,heat_index_weekly.loc["2020-01-05":])
bias = np.mean(forecast - heat_index_weekly.loc["2020-01-05":])
maxerror = max_error(forecast, heat_index_weekly.loc["2020-01-05":])
r_squared = r2_score( heat_index_weekly.loc["2020-01-05":],forecast)
run_time = end - start
print(f'Random Forest Regressor RMSE: {rmse}')
print(f'Random Forest Regressor MAE: {mae}')
print(f'Random Forest Regressor Bias: {bias}')
print(f'Random Forest Regressor Max Error: {maxerror}')
print(f'Random Forest Regressor R-Squared: {r_squared}')
print(f'Random Forest Regressor Run Time: {run_time}s')


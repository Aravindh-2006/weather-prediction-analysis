import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Simulate historical weather data (e.g., daily temperatures in °C)
np.random.seed(42)
days = 365  # Number of days
time = np.arange(days)
temperature = 25 + 10 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 1, days)

# Plot the simulated data
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, label="Temperature (°C)")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.title("Simulated Daily Temperature Data")
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(temperature) * 0.8)
train, test = temperature[:train_size], temperature[train_size:]

# Fit an ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) values
fitted_model = model.fit()

# Forecast on the test set
forecast = fitted_model.forecast(steps=len(test))
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(train)), train, label="Training Data")
plt.plot(np.arange(len(train), len(train) + len(test)), test, label="Test Data", color='orange')
plt.plot(np.arange(len(train), len(train) + len(forecast)), forecast, label="Forecast", color='green')
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Forecast Using ARIMA")
plt.legend()
plt.show()

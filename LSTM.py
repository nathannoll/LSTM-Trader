import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
LSTM = tf.keras.layers.LSTM

data = pd.read_csv("SPX.csv", parse_dates=["Date"], index_col="Date")
training_data = data.loc["2016":"2018", "Close"]
##training_data = data.loc["2010":"2015", "Close"]
##training_data = data.loc["1990":"2005", "Close"]

actual_data= data.loc["2019":"2020", "Close"]
##actual_data= data.loc["2018":"2020", "Close"]
##actual_data= data.loc["2015":"2020", "Close"]

training_data = training_data.asfreq('B').ffill()
actual_data = actual_data.asfreq('B').ffill()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_data = scaler.fit_transform(training_data.values.reshape(-1, 1))
scaled_actual_data = scaler.fit_transform(actual_data.values.reshape(-1, 1))

def create_sequences(data, sequence_length=60):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

sequence_length = 60

x_train, y_train = create_sequences(scaled_training_data, sequence_length)

x_test, y_test = create_sequences(scaled_actual_data, sequence_length)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

predicted_prices = model.predict(x_test)

predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"Mean Absolute Error: {mae}")

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Define the function to train and predict
def train_and_predict(training_range, testing_range, label):
    # Load the data
    data = pd.read_csv("C:/Users/Joel Carrasco/OneDrive/PRML/SPX.csv", parse_dates=["Date"], index_col="Date")

    # Extract training and testing data based on date ranges
    training_start, training_end = training_range
    testing_start, testing_end = testing_range
    
    # Slice the data based on the given date ranges
    training_data = data.loc[training_start:training_end, "Close"]
    actual_data = data.loc[testing_start:testing_end, "Close"]

    # Fill missing values and resample data
    training_data = training_data.asfreq('B').ffill()
    actual_data = actual_data.asfreq('B').ffill()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_training_data = scaler.fit_transform(training_data.values.reshape(-1, 1))
    scaled_actual_data = scaler.fit_transform(actual_data.values.reshape(-1, 1))

    # Create sequences for LSTM model
    def create_sequences(data, sequence_length=60):
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(x), np.array(y)

    sequence_length = 60
    x_train, y_train = create_sequences(scaled_training_data, sequence_length)
    x_test, y_test = create_sequences(scaled_actual_data, sequence_length)

    # Build the LSTM model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_prices, predictions)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title(f"Stock Price Prediction - {label}")
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    return mae, predictions

# Call the function for different horizons

# Short horizon (2016-2018 to predict 2019-2020)
mae_short, predictions_short = train_and_predict(("2016-01-01", "2018-12-31"), ("2019-01-01", "2020-12-31"), "Short Horizon (2016-2018 -> 2019-2020)")

# Medium horizon (2010-2015 to predict 2018-2020)
mae_medium, predictions_medium = train_and_predict(("2010-01-01", "2015-12-31"), ("2018-01-01", "2020-12-31"), "Medium Horizon (2010-2015 -> 2018-2020)")

# Long horizon (1990-2005 to predict 2015-2020)
mae_long, predictions_long = train_and_predict(("1990-01-01", "2005-12-31"), ("2015-01-01", "2020-12-31"), "Long Horizon (1990-2005 -> 2015-2020)")

# Print the Mean Absolute Errors
print(f"Short Horizon MAE: {mae_short}")
print(f"Medium Horizon MAE: {mae_medium}")
print(f"Long Horizon MAE: {mae_long}")

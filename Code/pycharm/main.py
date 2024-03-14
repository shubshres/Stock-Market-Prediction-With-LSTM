import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load preprocessed data into a DataFrame
data = pd.read_csv('../../data/tsla.csv')

train_data = data[:int(0.7 * len(data))]
test_data = data[int(0.7 * len(data)):]

# Reshape the training data into a 3D array for LSTM
X_train = train_data[['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(X_train.shape[0], 1, X_train.shape[1])

# Reshape the target variable into a 2D array
y_train = train_data['Daily_Return'].values.reshape(y_train.shape[0], 1)

# Reshape the testing data into a 3D array for LSTM
X_test = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(X_test.shape[0], 1, X_test.shape[1])

# Reshape the target variable into a 2D array
y_test = test_data['Daily_Return'].values.reshape(y_test.shape[0], 1)

# Create a sequential model
model = Sequential()

# Add an LSTM layer with 64 units and return sequences
model.add(LSTM(64, return_sequences=True))

# Add another LSTM layer with 32 units
model.add(LSTM(32))

# Add a fully connected layer with 1 output
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the testing data
model.evaluate(X_test, y_test)

# Predict the daily returns for the next 5 days
predicted_returns = model.predict(X_test[:5])

# Plot the daily returns
data.plot(x='Date', y='Daily_Return')

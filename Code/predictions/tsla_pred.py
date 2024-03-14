import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load preprocessed data into a DataFrame
data = pd.read_csv('./Data/preprocessed/tsla_preprocessed.csv')

# Use only the relevant columns for prediction
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']]

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Define the sequence length (number of time steps to consider)
sequence_length = 10

# Create sequences and labels
sequences = []
labels = []
for i in range(len(data_scaled) - sequence_length):
    seq = data_scaled[i:i + sequence_length]
    label = data_scaled[i + sequence_length, 5]  # Assuming 'Daily_Return' is the column to predict
    sequences.append(seq)
    labels.append(label)

# Convert sequences and labels to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Split the data into training and testing sets (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.3, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to get back to the original scale
predictions = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], predictions), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]

# Visualize the predictions against the actual values
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(predictions, label='Predicted Daily Return')
plt.plot(y_test, label='Actual Daily Return')
plt.legend()
plt.show()

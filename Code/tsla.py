import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the data
file_path = '../data/tsla.csv' 
tsla_data = pd.read_csv(file_path)

# Step 2: Handle missing values
tsla_data.dropna(inplace=True)

# Step 3: Select relevant features
selected_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
tsla_data = tsla_data[selected_features]

# Step 4: Convert the 'Date' to DateTime format
tsla_data['Date'] = pd.to_datetime(tsla_data['Date'])

# Step 5: Set 'Date' as Index
tsla_data.set_index('Date', inplace=True)

# Step 6: Optional - Feature Engineeringk
tsla_data['Daily_Return'] = tsla_data['Close'].pct_change()

# Step 7: Normalize numerical features
scaler = MinMaxScaler()
tsla_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']] = scaler.fit_transform(tsla_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']])

# Step 8: Save the preprocessed data
tsla_data.to_csv('../data/preprocessed/tsla_preprocessed.csv')  # Adjust the path accordingly

# Display the preprocessed data
print(tsla_data.head())

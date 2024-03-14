import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the data
file_path = '../data/aapl.csv'  
aapl_data = pd.read_csv(file_path)

# Step 2: Handle missing values
aapl_data.dropna(inplace=True)

# Step 3: Select relevant features
selected_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
aapl_data = aapl_data[selected_features]

# Step 4: Convert the 'Date' to DateTime format
aapl_data['Date'] = pd.to_datetime(aapl_data['Date'])

# Step 5: Set 'Date' as Index
aapl_data.set_index('Date', inplace=True)

# Step 6: Optional - Feature Engineering
aapl_data['Daily_Return'] = aapl_data['Close'].pct_change()

# Step 7: Normalize numerical features
scaler = MinMaxScaler()
aapl_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']] = scaler.fit_transform(aapl_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']])

# Step 8: Save the preprocessed data
aapl_data.to_csv('../data/preprocessed/aapl_preprocessed.csv')  # Adjust the path accordingly

# Display the preprocessed data
print(aapl_data.head())
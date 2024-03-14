import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the data
file_path = '../data/msft.csv'  
msft_data = pd.read_csv(file_path)

# Step 2: Handle missing values
msft_data.dropna(inplace=True)

# Step 3: Select relevant features
selected_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
msft_data = msft_data[selected_features]

# Step 4: Convert the 'Date' to DateTime format
msft_data['Date'] = pd.to_datetime(msft_data['Date'])

# Step 5: Set 'Date' as Index
msft_data.set_index('Date', inplace=True)

# Step 6: Optional - Feature Engineering
msft_data['Daily_Return'] = msft_data['Close'].pct_change()

# Step 7: Normalize numerical features
scaler = MinMaxScaler()
msft_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']] = scaler.fit_transform(msft_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']])

# Step 8: Save the preprocessed data
msft_data.to_csv('../data/preprocessed/msft_preprocessed.csv')  # Adjust the path accordingly

# Display the preprocessed data
print(msft_data.head())
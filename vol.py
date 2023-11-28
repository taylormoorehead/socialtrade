import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'S&P500Healthcare.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
df.set_index('Date', inplace=True)  # Set 'Date' as the index
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])
df = df[df.index <= '2020-04-15']

# Function to calculate historical volatility
def calculate_historical_volatility(data, lookback):
    returns = np.log(data['Close'] / data['Close'].shift(1))
    historical_volatility = returns.rolling(window=lookback).std() * np.sqrt(252)  # Assuming 252 trading days in a year
    return historical_volatility

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, lookback, multiplier):
    rolling_mean = data['Close'].rolling(window=lookback).mean()
    rolling_std = data['Close'].rolling(window=lookback).std()

    upper_band = rolling_mean + (rolling_std * multiplier)
    lower_band = rolling_mean - (rolling_std * multiplier)

    return upper_band, lower_band

lookback_period = 2
multiplier = 2.5

# Calculate historical volatility
df['HistoricalVolatility'] = calculate_historical_volatility(df, lookback_period)

# Calculate Bollinger Bands
df['UpperBand'], df['LowerBand'] = calculate_bollinger_bands(df, lookback_period, multiplier)

# Plotting the data and Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price', linewidth=1)
plt.plot(df['UpperBand'], label='Upper Bollinger Band', linestyle='--', linewidth=1)
plt.plot(df['LowerBand'], label='Lower Bollinger Band', linestyle='--', linewidth=1)

plt.title('Bollinger Bands with Historical Volatility')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

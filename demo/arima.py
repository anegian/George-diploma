"""
   Evaluate the given model on the test data and print classification report, 
   confusion matrix, and training score.

"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read the dataset
sales_data_df = pd.read_csv("sales.csv")

# Convert 'Year' and 'Month' columns to datetime format
sales_data_df['Date'] = pd.to_datetime(sales_data_df['Year'].astype(str) + '-' + sales_data_df['Month'].astype(str), format='%Y-%B')

# Set 'Date' column as index
sales_data_df.set_index('Date', inplace=True)

# Provide frequency information
sales_data_df.index.freq = 'MS'

# Prepare data
X = sales_data_df[['Sales', 'Predictions']]

# Split data into train and test sets
train_size = int(len(X) * 0.8)
train, test = X.iloc[:train_size], X.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train['Sales'], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Calculate mean squared error
mse = mean_squared_error(test['Sales'], forecast)
print("Mean Squared Error:", mse)

# Optionally, you can visualize the predictions compared to the actual values
plt.plot(test.index, test['Sales'], label='Actual')
plt.plot(test.index, forecast, label='Predicted')
plt.legend()
plt.show()

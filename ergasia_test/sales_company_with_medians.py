import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Load data
sales_data_df = pd.read_csv('sales.csv')

# Drop unnecessary columns
sales_data_df.drop("ID", axis=1, inplace=True)

# Convert Month to numerical values
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
sales_data_df['Month'] = sales_data_df['Month'].map(month_mapping)

# Prepare data
X = sales_data_df[['Year', 'Month']]
y = sales_data_df['Sales']

# Train the model
svr = SVR(kernel='rbf')
svr.fit(X, y)

# Calculate median of actual and predicted sales for each month
median_sales = sales_data_df.groupby('Month')['Sales'].median()
median_predictions = sales_data_df.groupby('Month')['Predictions'].median()
# Calculate ratio of medians
median_ratios = median_sales / median_predictions
print("\n ### MEDIAN SALES ###\n", median_sales)
print("\n ### MEDIAN PREDICTIONS ###\n", median_predictions)
print("\n ### MEDIAN RATIOS ###\n", median_ratios)

# Set threshold for deviation
THRESHOLD = 1.05

# # Calculate median deviation for each month
# historical_deviations = median_ratios - THRESHOLD
# print("\n ### HISTORICAL DEVIATIONS ###\n", historical_deviations)

# Apply median deviation to adjust predicted sales for 2025
adjusted_sales_forecast_2025 = []

for month in range(1, 13):
    forecast = svr.predict([[2025, month]])[0]
    # Use median ratio to adjust forecast
    if median_ratios[month] <= THRESHOLD:
        # If median ratio is within threshold, adjust the forecast
        adjusted_forecast = forecast * median_ratios[month]
        adjusted_sales_forecast_2025.append(adjusted_forecast)
    else:
        # Otherwise, use the original forecast
        adjusted_sales_forecast_2025.append(forecast * median_ratios[month])


# Create DataFrame for adjusted sales forecast for 2025
adjusted_forecast_df = pd.DataFrame({
    'Year': [2025] * 12,
    'Month': range(1, 13),
    'Adjusted_Sales': adjusted_sales_forecast_2025
})

print("\n ### FORECAST SALES 2025 ###\n", adjusted_forecast_df)

# Plot adjusted sales forecast for 2025
plt.figure(figsize=(10, 6))
plt.plot(adjusted_forecast_df['Month'], adjusted_forecast_df['Adjusted_Sales'], marker='o', color='b', linestyle='-')
plt.title('Sales Forecast for 2025')
plt.xlabel('Month')
plt.ylabel('Sales (boxes)')
plt.xticks(range(1, 13), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
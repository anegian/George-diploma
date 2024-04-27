"""
   Evaluate the given model on the test data and print classification report, 
   confusion matrix, and training score.

"""
# 1. Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import sys

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load data
sales_data_df = pd.read_csv('sales.csv')

# Drop unnecessary columns
sales_data_df.drop("ID", axis=1, inplace=True)

# Step 2: Feature Engineering
# Convert Month to numerical values
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
sales_data_df['Month'] = sales_data_df['Month'].map(month_mapping)

print(sales_data_df.head(96))

# Prepare your data
X = sales_data_df[['Year', 'Month', 'Predictions']]
y = sales_data_df['Sales']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
svr = SVR(kernel='rbf')
svr.fit(X_scaled, y)

# Forecasting for 2025
months_2025 = pd.DataFrame({'Year': [2025]*12, 'Month': range(1, 13), 'Predictions': 0})
months_2025_scaled = scaler.transform(months_2025)
sales_forecast_2025 = svr.predict(months_2025_scaled)
print("\n @@@@@ Sales Forecast 2025 @@@@@\n", sales_forecast_2025)

# Calculate historical percentage deviation for each month across multiple years
historical_deviations = []

for month in range(1, 13):
    sales_actual = sales_data_df[(sales_data_df['Month'] == month) & (sales_data_df['Year'] < 2025)]['Sales']
    sales_predicted = sales_data_df[(sales_data_df['Month'] == month) & (sales_data_df['Year'] < 2025)]['Predictions']
    deviation = (sales_actual / sales_predicted).median() - 1
    historical_deviations.append(deviation)
    print("DEVIATION ", month, ": ", deviation)

# Apply median deviation to adjust predicted sales for 2025
sales_forecast_adjusted_2025 = [forecast * (1 + deviation) for forecast, deviation in zip(sales_forecast_2025, historical_deviations)]

# Print the result of zip function
print("=== ZIP === ", list(zip(sales_forecast_2025, historical_deviations)))


# Filter the data for April across the years
april_data = sales_data_df[sales_data_df['Month'] == 3]
# Display the April data to inspect for any missing values
print("\n ===== April Data =====\n", april_data)

# Calculate the ratio between actual sales and predicted sales for April
april_data['Ratio'] = april_data['Sales'] / april_data['Predictions']

# Display the April data with calculated ratio
print("\n ===== April Data with Ratio =====\n", april_data)

# Calculate the median of the ratio between actual sales and predicted sales for April
median_deviation = (april_data['Sales'] / april_data['Predictions']).median()
median_deviation -= 1

print("\n ===== Median Deviation for April ====\n", median_deviation)


# Create a DataFrame for adjusted sales forecast for 2025
adjusted_forecast_df = pd.DataFrame({
    'Year': [2025] * 12,
    'Month': range(1, 13),
    'Predictions': sales_forecast_adjusted_2025,
    
})

print("\n ### FORECAST SALES 2025 ###\n", adjusted_forecast_df)

y_pred = svr.predict(adjusted_forecast_df)
print("\n ### SVR MODEL PREDICT 2025 ###\n", y_pred)

# Step 7: Analysis and Visualization
# Plot adjusted sales forecast for 2025
plt.figure(figsize=(10, 6))
plt.plot(adjusted_forecast_df['Month'], adjusted_forecast_df['Predictions'], marker='o', color='b', linestyle='-')
plt.title('Sales Forecast for 2025')
plt.xlabel('Month')
plt.ylabel('Sales (boxes)')
plt.xticks(range(1, 13), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

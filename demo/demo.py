"""
   Evaluate the given model on the test data and print classification report, 
   confusion matrix, and training score.

"""

import pandas as pd
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Read the dataset
training_data = pd.read_csv("sales.csv")
training_data_excel = pd.read_excel('book2.xlsx')

# # Drop unnecessary columns
# training_data.drop("ID", axis=1, inplace=True)
# Add a new column 'Ratio' containing the ratio of Sales to Predictions
training_data['Ratio'] = training_data['Sales'] / training_data['Predictions']
training_data['Differ'] = training_data['Sales'] - training_data['Predictions']

# print("=== TRAINING DATA ===\n")
# print(training_data.head(96))

# Step 2: Feature Engineering
# Convert Month to numerical values
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
training_data['Month'] = training_data['Month'].map(month_mapping)

# Convert X to numpy array
X = np.array(training_data[["Month", "Predictions", "Ratio", "Differ"]])
X_no_array = training_data[["Month", "Predictions", "Ratio", "Differ"]]
y = np.array(training_data["Sales"])
# print("=== TRAINING DATA X ===\n")
# print(X)
# print("=== TRAINING DATA X_no_array ===\n")
# print(X_no_array)

# # Shuffle the data
# training_data_shuffled = training_data.sample(frac=1, random_state=42)

#Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

# Step 6: Train the model
svr = SVR(kernel='linear')
svr.fit(X_train, y_train) # Exclude ID column from X_train

# Step 7: Get predictions for the test set
y_predict = svr.predict(X_test)


# # Print actual labels and predictions
# for actual, predicted in zip(y_test, y_predict):
#     print("Actual:", actual, "| Predicted:", predicted)

# Print actual labels, predictions, and the corresponding IDs
for idx, (ID, actual, predicted) in enumerate(zip(X_test[:, 0], y_test, y_predict), start=1):
    print("Row", idx, "| ID:", ID, "| Actual:", actual, "| Predicted:", predicted)

# # Create a DataFrame containing actual and predicted sales
# forecasts_df = pd.DataFrame({'Actual_Sales': y_test, 'Predicted_Sales': y_predict})

# # Write the DataFrame to an Excel file
# forecasts_df.to_excel('sales_forecasts.xlsx', index=False)

# print("Excel file 'sales_forecasts.xlsx' created successfully.")

# Calculate the average Ratio and Differ for each month from the training data
average_ratio_by_month = training_data.groupby('Month')['Ratio'].mean()
average_differ_by_month = training_data.groupby('Month')['Differ'].mean()

# Calculate the average predictions for each month from the training data
average_predictions_by_month = training_data.groupby('Month')['Predictions'].mean()


# Step 8: Create a DataFrame for adjusted sales forecast for 2025
predicted_sales_2025 = []
average_training_predictions = []
for month in range(1, 13):
    # Use the average prediction for the corresponding month as a placeholder
    average_prediction_for_month = average_predictions_by_month[month]
    average_training_predictions.append(average_prediction_for_month)
    average_ratio_for_month = average_ratio_by_month[month]
    average_differ_for_month = average_differ_by_month[month]
    predicted_sales_month = svr.predict([[month, average_prediction_for_month, average_ratio_for_month, average_differ_for_month] for _ in range(len(training_data))])
    predicted_sales_2025.append(predicted_sales_month[0])

# Create a DataFrame for the adjusted sales forecast for 2025
adjusted_forecast_df = pd.DataFrame({
    'Year': [2025] * 12,
    'Month': range(1, 13),
    'Predictions': predicted_sales_2025
})

# Print average predictions at training vertically
print("\n### AVERAGE PREDICTIONS AT TRAINING ###")
for month, prediction in zip(range(1, 13), average_training_predictions):
    print(f"Month: {month} | Prediction: {prediction}")

# Print forecast sales for 2025 vertically
print("\n### FORECAST SALES 2025 ###")
for month, forecast in zip(range(1, 13), predicted_sales_2025):
    print(f"Month: {month} | Forecast: {forecast}")

# Plot both average predictions and forecast sales on the same diagram
plt.figure(figsize=(10, 6))
plt.plot(range(1, 13), average_training_predictions, marker='o', color='g', linestyle='-', label='Average Training Predictions')
plt.plot(range(1, 13), predicted_sales_2025, marker='o', color='b', linestyle='-', label='Forecast Sales 2025')
plt.title('Average Training Predictions vs. Forecast Sales 2025')
plt.xlabel('Month')
plt.ylabel('Sales (boxes)')
plt.xticks(range(1, 13), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------

# # Filter the data for April across the years
# april_data = training_data[training_data['Month'] == 4]
# # Display the April data to inspect for any missing values
# print("\n ===== April Data =====\n", april_data)

# # Calculate the ratio between actual sales and predicted sales for April
# april_data['Ratio'] = april_data['Sales'] / april_data['Predictions']

# # Display the April data with calculated ratio
# print("\n ===== April Data with Ratio =====\n", april_data)

# # ----------------------------------------------

# # Calculate predicted sales for each month
# predicted_sales = []
# # actual_sales = []
# # month_ratio_array = []

# for month in range(1, 13):
#     # Filter the data for the specific month across the years
#     month_data = training_data[training_data['Month'] == month]

#     # Calculate the median of the actual sales for the specific month
#     predicted_sales_month = month_data['Predictions'].median()
# #     actual_sales_month = month_data['Sales'].median()
          
# #     # Calculate the ratio between actual sales and predicted sales for April
# #     month_ratio = month_data['Sales'] / month_data['Predictions']
  
#     predicted_sales.append(predicted_sales_month)
# #     actual_sales.append(actual_sales_month)
# #     month_ratio_array.append(month_ratio)

# # Convert predicted sales to a numpy array
# print("predicted_sales_array: \n", predicted_sales)
# # Convert the predicted_sales list to a 2D array
# predicted_sales_array = np.array(predicted_sales).reshape(-1, 1)
# print("predicted_sales_array: \n", predicted_sales_array)

# forecasts = svr.predict(predicted_sales_array)
# print("*** FORECASTS 2025 *** \n", forecasts)
# # print("actual_sales array: \n", actual_sales)

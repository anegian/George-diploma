"""
   Read training data set, train the model, forecast

"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Step 1: Read the dataset
training_data = pd.read_excel('training_data.xlsx')

# # Drop unnecessary columns
training_data.drop("Περιγραφή", axis=1, inplace=True)
training_data.drop("Κωδικός", axis=1, inplace=True)
# # Add a new column 'Ratio' containing the ratio of Sales to Predictions
# training_data['Ratio'] = training_data['Sales'] / training_data['Forecasts']
# training_data['Differ'] = training_data['Sales'] - training_data['Forecasts']

# Step 2: Feature Engineering
# Convert Month to numerical values
month_mapping = {
    "Ιανουάριος": 1, "Φεβρουάριος": 2, "Μάρτιος": 3, "Απρίλιος": 4,
    "Μάιος": 5, "Ιούνιος": 6, "Ιούλιος": 7, "Αύγουστος": 8,
    "Σεπτέμβριος": 9, "Οκτώβριος": 10, "Νοέμβριος": 11, "Δεκέμβριος": 12
}
training_data['Μήνας'] = training_data['Μήνας'].map(month_mapping)

print("=== TRAINING DATA ===\n")
print(training_data)

# Convert X to numpy array
X = np.array(training_data[["Μήνας", "Forecasts"]])
y = np.array(training_data["Sales"])

#Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

# Step 6: Train the model
svr = SVR(kernel='linear')
svr.fit(X_train, y_train) # Exclude ID column from X_train

# Step 7: Get predictions for the test set
y_predict = svr.predict(X_test)

# Print actual labels, predictions, and the corresponding IDs
for idx, (ID, actual, predicted) in enumerate(zip(X_test[:, 0], y_test, y_predict), start=1):
    print("Row", idx, "| ID:", ID, "| Actual:", actual, "| Predicted:", predicted)

# Calculate the average predictions for each month from the training data
average_predictions_by_month = training_data.groupby('Μήνας')['Forecasts'].mean()


# Step 8: Create a DataFrame for adjusted sales forecast for 2025
predicted_sales_2025 = []
average_training_predictions = []
for month in range(1, 13):
    # Use the average prediction for the corresponding month as a placeholder
    average_prediction_for_month = average_predictions_by_month[month]
    average_training_predictions.append(average_prediction_for_month)
    predicted_sales_month = svr.predict([[month, average_prediction_for_month] for _ in range(len(training_data))])
    predicted_sales_2025.append(predicted_sales_month[0])

# Create a DataFrame for the adjusted sales forecast for 2025
adjusted_forecast_df = pd.DataFrame({
    'Year': [2025] * 12,
    'Μήνας': range(1, 13),
    'Forecasts': predicted_sales_2025
})

# Print average predictions at training vertically
print("\n### AVERAGE PREDICTIONS AT TRAINING ###")
for month, prediction in zip(range(1, 13), average_training_predictions):
    print(f"Μήνας: {month} | Forecasts: {prediction}")

# Print forecast sales for 2025 vertically
print("\n### FORECAST SALES 2025 ###")
for month, forecast in zip(range(1, 13), predicted_sales_2025):
    print(f"Μήνας: {month} | Forecasts: {forecast}")

# Plot both average predictions and forecast sales on the same diagram
plt.figure(figsize=(10, 6))
plt.plot(range(1, 13), average_training_predictions, marker='o', color='g', linestyle='-', label='Average Training Predictions')
plt.plot(range(1, 13), predicted_sales_2025, marker='o', color='b', linestyle='-', label='Forecast Sales 2025')
plt.title('Average Training Predictions vs. Forecast Sales 2025')
plt.xlabel('Μήνας')
plt.ylabel('Sales (boxes)')
plt.xticks(range(1, 13), ['Ιανουάριος', 'Φεβρουάριος', 'Μάρτιος', 'Απρίλιος', 'Μάιος', 'Ιούνιος', 'Ιούλιος', 'Αύγουστος', 'Σεπτέμβριος', 'Οκτώβριος', 'Νοέμβριος', 'Δεκέμβριος'], rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
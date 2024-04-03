"""
   Evaluate the given model on the test data and print classification report, 
   confusion matrix, and training score.

"""
# 1. Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Data from csv file for training the model
sales_training = pd.read_csv('C:\\Users\\user\\Desktop\\ergasia\\sales_prediction.csv')

# Take only the first 100 rows for training
X_train = sales_training.drop(["Month"], axis=1)
y_train = sales_training["Month"]

# Modeling (SVM with scikit-learn)
svc_model = SVC(kernel="linear", gamma="scale", C=1.0)
svc_model.fit(X_train, y_train)

# Load the actual sales data for comparison
actual_sales = pd.read_csv('C:\\Users\\user\\Desktop\\ergasia\\sales.csv')

# Extract the actual months and quantities sold
actual_sales["Month"] = pd.to_datetime(actual_sales["Month"])

# Predict quantities sold for the years 2010-2018
X_test_years = actual_sales.drop(["Month"], axis=1)
predicted_quantities_sold = svc_model.predict(X_test_years)

# Plotting actual vs predicted sales for the years 2010-2018
plt.figure(figsize=(10, 6))
plt.plot(actual_sales.index, actual_sales["Quantity Sold (kg)"], label="Actual Sales")
plt.plot(actual_sales.index, predicted_quantities_sold, label="Predicted Sales")
plt.xlabel("Year")
plt.ylabel("Quantity Sold (kg)")
plt.title("Actual vs Predicted Sales (2010-2018)")
plt.legend()
plt.grid(True)
plt.show()

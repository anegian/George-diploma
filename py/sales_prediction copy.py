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
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report


# 2. Load Data from csv file
sales_df = pd.read_csv('C:\\Users\\user\\Desktop\\ergasia\\supermarket-sales.csv')

print("\n=== SALES TABLE (first 5 rows) === \n",sales_df.head())
print("\n=== SALES TABLE (last 5 rows) === \n",sales_df.tail())
print("\n### Shape: ###",sales_df.shape)
print("\n### Size ###: ", sales_df.size)
print("\n### Count: ###",sales_df.count())
print(sales_df["Gender"].value_counts())
# Prints if the column has numerical content or objects. Objects cannot be compared on a plot
print("\n### D-Types ###: ", sales_df.dtypes)
print(sales_df.columns)

# # Identifying unwanted rows
# sales_df = sales_df[pd.to_numeric(sales_df["Product Category"], errors="coerce").notnull()]
# sales_df["Product Category"] = sales_df["Product Category"].astype('int')
# print("\n### D-Types ###: ", sales_df.dtypes)

# picked only 3 columns
filtered_sales_df = sales_df.drop(["Product ID","Product Category","Gender"], axis=1)

# Independent variable
X=np.array(filtered_sales_df)

# Dependent variable
y= np.array(sales_df["Gender"])

print(X)
print(y)

# Divide the data as Train/Test dataset:
"""
sales_df(10) --> Train (8 rows) / Test (2 rows)
Train(X, y) ## X itself is a 2D array. ##y is 1D
Test(X,y)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Modeling (SVM with scikit-learn)
# SVC (Support Vector Classifier)
# Kernel choices: 1. Linear, 2. Polynomial, 3. RBF (Radial basis function), 4. Sigmoid

svc_model = SVC(kernel="linear", gamma="scale", C=1.0)
svc_model.fit(X_train, y_train)

y_predict = svc_model.predict(X_test)

# Print actual labels and predictions
for actual, predicted in zip(y_test, y_predict):
    print("Actual:", actual, "| Predicted:", predicted)

# Check if there are any instances of "Male" in the test set
if "Male" not in y_test:
    print("No instances of 'Male' in the test set")

# Check if there are any instances of "Male" in the predictions
if "Male" not in y_predict:
    print("No instances of 'Male' in the predictions")


# Evaluate (Results)
print("\n ** REPORT **\n", classification_report(y_test, y_predict))

#Distribution of classes
male_df = sales_df[sales_df["Gender"]=='Male']
female_df = sales_df[sales_df["Gender"]=='Female']

axes = male_df.plot(kind="scatter", x="Quantity", y="Total", color="blue", label="Male")
female_df.plot(kind="scatter", x="Quantity", y="Total", color="red", label="Female", ax=axes)

plt.show()
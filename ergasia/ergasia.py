"""
   Evaluate the given model on the test data and print classification report, 
   confusion matrix, and training score.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load data
market = pd.read_csv('supermarket-sales.csv')

# Drop unnecessary columns
market.drop("Product ID", axis=1, inplace=True)  # Corrected column name

# Encode categorical variables
label_encoder = LabelEncoder()
for col in market.select_dtypes(include=['object']).columns:
    market[col] = label_encoder.fit_transform(market[col])

# Split data into features and target variable
X = market.drop("Gender", axis=1)
y = market["Gender"]

# Split data into training and testing sets
# test_size=0.2, means that 20% of the data will be reserved for testing, 
#and the remaining 80% will be used for training the model.
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Function to evaluate model and print results
def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate the given model on the test data and print classification report, 
    confusion matrix, and training score.

    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division='warn'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Training Score:", model.score(x_train, y_train) * 100)
    if isinstance(model, SVC):
        # Calculate and print accuracy score
        acc_score = accuracy_score(y_test, y_pred)
        print("Accuracy Score:", acc_score * 100)

        # Calculate and print mean squared error
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse * 100)

# Initialize models
classifiers = {
    "SVC": SVC(),
    "Random Forest Classifier": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, random_state=123, criterion='entropy'),
    "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=100, random_state=0)
}

# Evaluate each model
for name, model in classifiers.items():
    print("\n" + "="*20 + f" {name} " + "="*20)
    evaluate_model(model, x_train, y_train, x_test, y_test)
    # plt.style.use("ggplot")  # Apply the desired style here
    # plt.scatter(x_test,y_test)
    # plt.show()

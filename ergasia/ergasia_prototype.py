# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:19:01 2024

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

market = pd.read_csv('supermarket-sales.csv')
market.isna().sum()

market.drop("Product ID", axis=1,inplace=True)

Col_list = list(market.columns)

Cat_list = []
for col in Col_list:
    if market[col].dtype == "object":
        Cat_list.append(col)

from sklearn.preprocessing import LabelEncoder

#used to transform non-numerical labels 
#(as long as they arehashable and comparable) to numerical labels.
L_E = LabelEncoder()

for i in Cat_list:
    market[i] = L_E.fit_transform(market[i])
# fit_transform: Fit label encoder and return encoded labels.

X = market.drop("Gender",axis=1)

y = market["Gender"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size =0.2)

from sklearn.svm import SVC #supervised learning methods used for classification, regression and outliers detection.
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import mean_squared_error

svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

accuracy_score(y_test,y_pred)*100

confusion_matrix(y_test,y_pred)

mean_squared_error(y_test,y_pred)*100

print(svc.score(x_train,y_train)*100)

from sklearn.ensemble import RandomForestClassifier
R_F_C = RandomForestClassifier()
R_F_C.fit(x_train,y_train)
y_pred1 = R_F_C.predict(x_test)
confusion_matrix(y_test,y_pred1)
mean_squared_error(y_test,y_pred1)*100
print(R_F_C.score(x_train,y_train)*100)


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",knn.score(x_train,y_train)*100)

#SVC
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",svc.score(x_train,y_train)*100)


# gaussin
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gnb.score(x_train,y_train)*100)

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)

y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)

# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)

# ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
etc.fit(x_train,y_train)

y_pred=etc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",etc.score(x_train,y_train)*100)
# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start.
2. Import libraries and load the dataset.
3. Preprocess data and select features.
4. Split data into training and testing sets.
5. Initialize weights, bias and learning rate.
6. Apply sigmoid function and Gradient Descent.
7. Predict placement status, evaluate accuracy and stop. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Shanmuga priya K
RegisterNumber: 212225040401
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('/Placement_Data (1).csv')

data = data.drop(['sl_no', 'salary'], axis=1)

data = pd.get_dummies(data, drop_first=True, dtype=int)

X = data.drop('status_Placed', axis=1).values
y = data['status_Placed'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m = np.zeros(X_train.shape[1])
c = 0
learning_rate = 0.05
epochs = 10000
n = len(y_train)

for i in range(epochs):

    z = np.dot(X_train, m) + c
    y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    dm = np.dot(X_train.T, (y_pred - y_train)) / n
    dc = np.sum(y_pred - y_train) / n

    m = m - learning_rate * dm
    c = c - learning_rate * dc

z = np.dot(X_test, m) + c
y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

y_pred = (y_pred >= 0.5).astype(int)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

if y_pred[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed") 
```

## Output:


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


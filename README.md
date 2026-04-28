# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import required libraries (NumPy, Pandas, Matplotlib).
2. Load the dataset from Startup.csv,Select input (R&D Spend) and output (Profit),Convert data into NumPy arrays.
3. Normalize the input data,Initialize slope m = 0 and intercept b = 0,Set learning rate and number of iterations (epochs).
4. Repeat for many iterations: --> Predict values using y = mx + b --> Find error between actual and predicted values --> Calculate changes for m and b --> Update m and b to reduce error --> After training, get final values of m and b --> Use final equation to predict profit --> Plot actual data and best-fit line on graph

## Program:
```.py
/*
Program to implement the linear regression using gradient descent.
Developed by: Srinithi muthukumar
RegisterNumber:  212224240161
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Startup.csv")

X = data['R&D Spend'].values
y = data['Profit'].values

X = (X - X.mean()) / X.std()

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    y_pred = m * X + b
    
    
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

y_pred = m * X + b

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
```

## Output:
<img width="1047" height="721" alt="image" src="https://github.com/user-attachments/assets/da363df6-a90e-4745-9dec-74b7796f08e3" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

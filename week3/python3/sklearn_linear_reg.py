from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np 

data = pd.read_csv("ex1data1.txt")
X = list(data[data.columns[0]])
y = list(data[data.columns[1]])

n_samples = len(X)

for i in range(n_samples):
    X[i] = np.array(X[i])
    y[i] = np.array(y[i])

X = np.array(X)
y = np.array(y)
linear_regression  = LinearRegression()  # add params
linear_regression.fit(X,y)


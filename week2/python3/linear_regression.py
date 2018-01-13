import pandas as pd
import numpy as np
import matplotlib as plt    # red

def gradient_descent(X,y,theta,n_iterations,n_samples,alpha=0.01):
    for i in range(n_iterations):
        h = np.dot(X,theta)
        theta = theta - alpha*(1/n_samples)*np.dot(X.transpose(),(h-y))
    print("Theta : ", theta)
    predictions = np.dot(X,theta)
    sqError = np.square(predictions-y)
    cost = (1/(2*n_samples))*sum(sqError)
    print("Cost : ", cost) 
    return theta


df =  pd.read_csv('../ex1data1.txt',header=None,delimiter=',')

n_features = len(df.columns)
n_samples = len(df.index)
n_iterations = 1500

X = np.array(df[df.columns[0:-1]]).reshape((n_samples,n_features-1))
y = np.array(df[n_features-1]).reshape((n_samples,1))

one = np.ones(n_samples).reshape((n_samples,1))
X =np.concatenate((one,X),axis=1)

theta = np.zeros((n_features,1))

gradient_descent(X,y,theta,n_iterations,n_samples)

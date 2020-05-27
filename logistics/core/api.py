import numpy as np

# Core functions for logistics regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    return X @ theta

def cost(theta,X,y):
    m,n = X.shape
    hx = sigmoid(hypothesis(X,theta.reshape(len(theta),1)))
    return np.sum((-y * np.log(hx)) - ((1-y) * np.log(1-hx))) * 1/m

def gradient(theta, X,y):
    m,n = X.shape
    hx = sigmoid(hypothesis(X,theta.reshape(len(theta),1)))
    g = ((hx - y).T @ X) * 1/m
    return g.flatten() if theta.ndim==1 else g
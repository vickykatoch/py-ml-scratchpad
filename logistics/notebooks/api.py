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

def map_features(x1,x2,degree):
    features = np.ones((len(x1),1))
    for i in range(1,degree+1):        
        for j in range(0,i+1):
            z = np.power(x1 , (i-j)) * np.power(x2 , j)
            features = np.hstack([features,z])
    return features

def cost_regularized(theta, X, y, lmbda):
    m,n = X.shape
    n_cost = cost(theta,X,y)
    n_cost += np.sum((theta[2:len(theta):] ** 2)) * (lmbda/(2*m))
    return n_cost

def gradient_regularized(theta,X,y,lmbda):
    grad = gradient(theta,X,y)
    m,n = X.shape
    return grad + np.hstack([[[0]], ((lmbda/m) *  theta[1:,:]).T])
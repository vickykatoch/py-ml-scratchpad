import numpy as np
import os

def computeCost(X, y, theta):
    m = len(X)
    costs = np.power(np.matmul(X, theta) - y, 2)
    return np.sum(costs) / (2*m)


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(X)
    cost = np.zeros((iterations,2))    
    for i in range(0, iterations):
        h = np.matmul(X, theta) - y
        sm = np.sum(np.multiply(h, X), axis=0)
        z = np.multiply(sm, alpha*1/m)
        theta = (theta.T - z).T
        cost[i,0]=i
        cost[i,1] = computeCost(X, y, theta)
    return [cost, theta]

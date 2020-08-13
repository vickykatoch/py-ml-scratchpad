import numpy as np

def computeCost(X, y, theta):
    m = len(X)
    costs = np.power(np.matmul(X, theta) - y, 2)
    return np.sum(costs) / (2*m)
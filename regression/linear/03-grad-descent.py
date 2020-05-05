import numpy as np
from matplotlib import pyplot as plt

# region COMMON FUNCS


def computeCost(X, y, theta, m):
    costs = np.power(np.matmul(X, theta) - y, 2)
    return np.sum(costs) / (2*m)


def gradient(X, y, theta, alpha, iterations):
    m = len(X)
    cost = np.zeros(iterations)
    for i in range(0, iterations):
        h = np.matmul(X, theta) - y
        sm = np.sum(np.multiply(h, X), axis=0)
        z = np.multiply(sm, alpha*1/m)
        theta = (theta.T - z).T
        cost[i] = computeCost(X, y, theta, m)
    return [cost, theta]


def plotData(x, y, h, xlabel, ylabel, xlim, ylim):
    plt.subplots()
    plt.plot(X, h)
    plt.scatter(x, y, c='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(color='gray', linestyle='--', linewidth=.6,
             axis='both', which='both', alpha=.4)
    plt.show()

# endregion


data = np.genfromtxt('../../data/food-cart.csv', delimiter=',')
m = len(data)
X = np.hstack([np.ones([m, 1]), data[:, 0].reshape(m, 1)])
y = data[:, 1].reshape(m, 1)
iterations = 1500
alpha = 0.01
initial_theta = np.array([[0], [0]])

[costs, thetas] = gradient(X, y, initial_theta, alpha, iterations)
predictions = np.matmul(X, thetas)
plotData(X[:, 1], y, predictions, 'City Population (10K)',
         'Profits ($10K)', [4, 24], [-5, 25])
print(thetas)

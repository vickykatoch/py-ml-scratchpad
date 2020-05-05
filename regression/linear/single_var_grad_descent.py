import numpy as np
from matplotlib import pyplot as plt
from os import path
import regression.api.gradient as grad
import regression.api.visualize as viz


def run():
    data_file = path.join(path.dirname(__file__),'../../data/food-cart.csv')
    data = np.genfromtxt(data_file, delimiter=',')
    m = len(data)
    X = np.hstack([np.ones([m, 1]), data[:, 0].reshape(m, 1)])
    y = data[:, 1].reshape(m, 1)
    iterations = 2000
    alpha = 0.01
    initial_theta = np.array([[0], [0]])

    [costs, thetas] = grad.gradient_descent(X, y, initial_theta, alpha, iterations)
    print(thetas)
    print(costs)
    predictions = np.matmul(X, thetas)
    viz.plotData(X[:, 1], y, predictions, 'City Population (10K)',
            'Profits ($10K)', [4, 24], [-5, 25])
    
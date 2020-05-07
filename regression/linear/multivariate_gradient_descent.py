import regression.api.feature_normalize as normalize
import regression.api.gradient as grad
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import regression.helpers.data_helper as data_helper


def _visualize(costX, costY, dataX, dataY, predictions, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Gradient Descent Single Variable ({})'.format(title))
    ax1.plot(dataX, predictions)
    ax1.set_xlabel('City Population (10K)')
    ax1.set_ylabel('Profits ($10K)')
    ax1.scatter(dataX, dataY, c='red')
    ax1.legend()
    ax2.plot(costX, costY, c='red')
    ax2.set_title('Cost Function')
    ax2.set_ylabel('Cost')
    ax2.set_xlabel('Iterations')
    ax2.legend('Cost Curve')
    plt.show()


def run():
    data = data_helper.getHousesData()
    m = len(data)
    house_size = data[:, 0].reshape(m, 1)
    bed_rooms = data[:, 1].reshape(m, 1)
    price = data[:, 2].reshape(m, 1)
    initial_theta = np.zeros((3, 1))
    iterations = 400
    alpha = 0.01

    hs = normalize.normalize_meanbystd(house_size)
    print(hs)

    X = np.hstack([np.ones((m, 1)), normalize.normalize_meanbystd(
        house_size), normalize.normalize_meanbystd(bed_rooms)])
    [cost, thetas] = grad.gradient_descent(
        X, price, initial_theta, alpha, iterations)
    print(thetas)
    predictions = np.matmul(X, thetas)
    _visualize(cost[:, 0], cost[:, 1], X[:, 1], price, predictions, 'House')

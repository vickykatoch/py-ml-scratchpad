import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import path
import regression.api.gradient as grad
from sklearn.model_selection import train_test_split


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

    data_file = path.join(path.dirname(__file__), '../../data/weather.csv')
    df = pd.read_csv(data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        df['MinTemp'], df['MaxTemp'], test_size=0.2, random_state=0)

    m = len(X_train)
    X_train = X_train.to_numpy().reshape(m, 1)
    y_train = y_train.to_numpy().reshape(m, 1)
    X = np.hstack([np.ones([m, 1]), X_train[:, 0].reshape(m, 1)])
    iterations = 100
    alpha = 0.01
    initial_theta = np.array([[0], [0]])
    [costs, thetas] = grad.gradient_descent(
        X, y_train, initial_theta, alpha, iterations)
    title = "Slope : {}, Y Intercept : {}".format(thetas[0, 0], thetas[1, 0])
    predictions = np.matmul(X, thetas)
    _visualize(costs[:, 0], costs[:, 1], X[:, 1], y_train, predictions, title)

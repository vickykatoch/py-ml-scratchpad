import regression.api.feature_normalize as normalize
import regression.api.gradient as grad
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import regression.helpers.data_helper as data_helper


def _visualize(costX, costY, dataX, dataY, predictions, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Gradient Descent Single Variable ({})'.format(title))
    ax1.plot(dataX, predictions,c='blue')
    ax1.set_xlabel('City Population (10K)')
    ax1.set_ylabel('Profits ($10K)')
    ax1.plot(dataX, dataY, c='red')
    ax1.legend()
    
    ax2.plot(costX, costY, c='red')
    ax2.set_title('Cost Function')
    ax2.set_ylabel('Cost')
    ax2.set_xlabel('Iterations')
    ax2.legend('Cost Curve')
    plt.show()

def predict(X, mu, sigma, theta):
    norm_x = (X - mu)/sigma
    z = np.matmul(np.hstack([1, norm_x]) , theta)
    return z

def run():
    data = data_helper.getHousesData()
    m = len(data)
    house_size = data[:, 0].reshape(m, 1)
    bed_rooms  = data[:, 1].reshape(m, 1)
    price = data[:, 2].reshape(m, 1)
    initial_theta = np.zeros((3, 1))
    iterations = 400
    alpha = 0.01

    X = np.hstack([house_size,bed_rooms])
    (mu, sigma, norm_x) = normalize.normalize_stdsample(X)
    X = np.hstack([np.ones((m, 1)),norm_x])
    [cost, thetas] = grad.gradient_descent(X, price, initial_theta, alpha, iterations)
    # Predict the price for 1650sqft & 3 bedroom house
    p_price = predict([1650,3], mu, sigma, thetas)
    
    print("Theta : {}, Predicted price for 1650sqft with 3 bed rooms house : {}".format(thetas.T,p_price))
    y = np.matmul(X, thetas)
    
    _visualize(cost[:, 0], cost[:, 1], X[:, 1], price, y, 'House')

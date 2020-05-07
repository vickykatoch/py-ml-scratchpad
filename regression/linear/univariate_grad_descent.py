import numpy as np
from matplotlib import pyplot as plt
from os import path
import regression.api.gradient as grad

def _visualize(costX, costY, dataX, dataY, predictions, title):
    fig, (ax1, ax2) = plt.subplots(1,2)    
    ax1.set_title('Gradient Descent Single Variable ({})'.format(title)) 
    ax1.plot(dataX,predictions)
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
    data_file = path.join(path.dirname(__file__),'../../data/food-cart.csv')
    data = np.genfromtxt(data_file, delimiter=',')
    m = len(data)
    X = np.hstack([np.ones([m, 1]), data[:, 0].reshape(m, 1)])
    y = data[:, 1].reshape(m, 1)
    iterations = 2000
    alpha = 0.01
    initial_theta = np.array([[0], [0]])

    [costs, thetas] = grad.gradient_descent(X, y, initial_theta, alpha, iterations)
    title = "Slope : {}, Y Intercept : {}".format(thetas[0,0],thetas[1,0])
    predictions = np.matmul(X, thetas)
    _visualize(costs[:,0], costs[:,1],X[:, 1], y, predictions,title)
   
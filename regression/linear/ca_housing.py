import numpy as np
import pandas as pd
import regression.helpers.data_helper as data_helper
import matplotlib.pyplot as plt

def visualize(epochs,costs):
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.suptitle('Cost by epoch')
    plt.plot(epochs,costs, linewidth=1)
    plt.show()

def bias_coef_update(m, b, X, Y, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = len(Y)
    # iterate over examples
    for idx in range(len(Y)):
        x = X[idx]
        y = Y[idx]
        # predict y with current bias and coefficient
        y_hat = (m * x) + b
        m_gradient += -(2/N) * x * (y - y_hat)
        b_gradient += -(2/N) * (y - y_hat)
    # use gradient with learning_rate to nudge bias and coefficient
    new_coef = m - (m_gradient * learning_rate)
    new_bias = b - (b_gradient * learning_rate)
    return new_coef, new_bias

def cost(x,y,m,b):
    return (m*x + b) - y

def  start(epoch_count=1000):
    df = pd.read_csv(data_helper.getCaliforniaHousingCsv())
    # Reduce the Data size by 75%
    df = df.sample(frac=0.25)
    # Collect our input features and labels
    X = df['median_income'].tolist()
    y = df['median_house_value'].tolist()

    # store output to plot later
    epochs = []
    costs = []
        
    m = 0 
    b = 0 
    learning_rate = 0.01
    for i in range(epoch_count):
        m, b = bias_coef_update(m, b, X, y, learning_rate)
        # print(m,b)
        C = cost(X[i], y[i], m, b)
        # C = cost(b, m, x_y_pairs)
        
        epochs.append(i)
        costs.append(C)
    
    return epochs, costs, m, b

def run():
    epochs, costs, m, b = start()
    print(m)
    print(b)
    print(costs[-1])
    visualize(epochs,costs)


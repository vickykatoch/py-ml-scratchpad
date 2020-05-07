import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import path

def plotData(x, y, h,title):
    plt.subplots()
    plt.plot(x, h)
    plt.scatter(x, y, c='red')
    plt.grid(color='gray', linestyle='--', linewidth=.6,
             axis='both', which='both', alpha=.4)
    plt.title(title)
    plt.show()


def calcR2(actual_y, predicted_y, avg_y):
    return np.square(predicted_y - avg_y).sum()/np.square(actual_y - avg_y).sum()


def calcSlope(x, y, mean_x, mean_y):
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum(np.square(x - mean_x))
    return numerator/denominator

def run():
    data_file = path.join(path.dirname(__file__),'../../data/food-cart.csv')
    data = np.genfromtxt(data_file, delimiter=',')
    X = data[:,0]
    Y = data[:,1]
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    slope_m = calcSlope(X, Y, mean_x, mean_y)
    intercept_c = mean_y - mean_x * slope_m
    m = len(X)
    H = []
    for i in range(0, m):
        H.append(slope_m * X[i] + intercept_c)
    title = "Slope : {}, Y Intercept : {}".format(slope_m, intercept_c)
    print("Slope : {}, Y Intercept : {}, Prediction Accuracy (R2 Error) : {}".format(
        slope_m, intercept_c, calcR2(Y, H, mean_y)))
    plotData(X, Y, H,title)
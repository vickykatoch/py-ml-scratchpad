import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plotData(x, y, h):
    plt.subplots()
    plt.plot(x, h)
    plt.scatter(x, y, c='red')
    plt.grid(color='gray', linestyle='--', linewidth=.6, axis='both', which='both', alpha=.4)
    plt.show()

def calcR2(actual_y,predicted_y,avg_y):
    return np.square(predicted_y - avg_y).sum()/np.square(actual_y - avg_y).sum()

X = np.array([1,2,3,4,5])
Y = np.array([3,4,2,4, 5])

mean_x = np.mean(X)
mean_y = np.mean(Y)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum(np.square(X - mean_x))
slope_m = numerator/denominator
intercept_c = mean_y - mean_x * slope_m
m = len(X)
H=[]
for i in range(0,m):
    H.append(slope_m * X[i] + intercept_c)
print("Slope : {}, Y Intercept : {}, Prediction Accuracy (R2 Error) : {}".format(slope_m,intercept_c,calcR2(Y,H,mean_y)))  
plotData(X,Y,H)
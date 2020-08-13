import numpy as np
from os import path

def calcZScore(data):
    # # of std deviation away from mean
    mn = np.mean(data)
    sd = np.std(data)
    return mn - data/sd

file_name = path.join(path.dirname(__file__),'../../data/housing-data.csv')


data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
x = data[:,0]
y = data[:,1]
s_size = len(x)
x_zsore = calcZScore(x)
y_zscore = calcZScore(y)
x_mean = np.mean(x)
y_mean = np.mean(y)

cor_coff = (x_zsore * y_zscore).sum()/s_size

print(cor_coff)


# y = mx + b
# m = corelation_coff * y_std/x_std

# Correlation Coefficent
# 1/observations * (x_zscore * y_zscore).sum()



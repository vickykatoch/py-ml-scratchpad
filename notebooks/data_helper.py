import numpy as np
from os import path



def getHousesData():
    return np.genfromtxt('../data/houses.csv', delimiter=',')

def getTestScroresData():
    return np.genfromtxt('../data/test_scores.csv', delimiter=',')

def getFoodCartData():
    return np.genfromtxt('../data/food-cart.csv', delimiter=',')
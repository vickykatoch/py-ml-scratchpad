import numpy as np
from os import path



def getHousesData():
    data_file = path.join(path.dirname(__file__),'../../data/houses.csv')
    return np.genfromtxt(data_file, delimiter=',')

def getTestScroresData():
    data_file = path.join(path.dirname(__file__),'../../data/test_scores.csv')
    return np.genfromtxt(data_file, delimiter=',')

def getFoodCartData():
    data_file = path.join(path.dirname(__file__),'../../data/food-cart.csv')
    return np.genfromtxt(data_file, delimiter=',')

def getCaliforniaHousingCsv():
    return path.join(path.dirname(__file__),'../../data/ca-housing.csv')
    
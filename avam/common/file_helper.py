from os import path


def getHousingDataFile():
    return path.join(path.dirname(__file__),'../../data/housing.data')
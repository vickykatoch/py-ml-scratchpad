import unittest
import numpy as np
from os import path
from regression.api.cost import computeCost



class CostFunctionTestCase(unittest.TestCase):
    """Tests for `cost.py`"""

    def setUp(self):
        self.EXPECTED_OUTPUT = np.round(32.072733877455676,8)
        data_file = path.join(path.dirname(__file__),'../../data/food-cart.csv')
        data = np.genfromtxt(data_file, delimiter=',')
        self.m = len(data)
        self.y = data[:, 1].reshape(self.m, 1)
        self.X = np.hstack([np.ones([self.m, 1]), data[:, 0].reshape(self.m, 1)])
        self.theta = np.array([[0], [0]])
    
    def testCost(self):
        cost = np.round(computeCost(self.X,self.y,self.theta),8)
        self.assertTrue(cost == self.EXPECTED_OUTPUT)

if __name__ == '__main__':
    unittest.main()

# python -m unittest    
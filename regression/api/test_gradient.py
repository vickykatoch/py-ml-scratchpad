import unittest
import numpy as np
from os import path
import regression.api.gradient as gd



class GradientDescentTestCase(unittest.TestCase):
    """Tests for `gradient.py`"""
    def setUp(self):
        data_file = path.join(path.dirname(__file__),'../../data/food-cart.csv')
        data = np.genfromtxt(data_file, delimiter=',')
        self.m = len(data)
        self.y = data[:, 1].reshape(self.m, 1)
        self.X = np.hstack([np.ones([self.m, 1]), data[:, 0].reshape(self.m, 1)])
        self.initial_theta = np.array([[0], [0]])
        self.iterations = 2000
        self.alpha = 0.01

    def testGradientDescent(self):
        [costs, thetas] = gd.gradient_descent(self.X, self.y, self.initial_theta, self.alpha, self.iterations)
        self.assertTrue(len(costs)==2000)
        self.assertTrue(len(thetas)==2)

if __name__ == '__main__':
    unittest.main()

# python -m unittest
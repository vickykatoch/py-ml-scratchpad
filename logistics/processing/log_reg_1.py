import numpy as np
import matplotlib.pyplot as plt
import logistics.core.api as api
from os import path
import scipy.optimize as op

data = np.genfromtxt(path.join(path.dirname(__file__),'../data/raw/ex2data1.txt'),delimiter=",")
X = data[:,0:2]
y = data[:,2:]
m = len(X)
X = np.hstack([np.ones((m,1)),X])
m , n = X.shape
initial_theta = np.zeros((n,1))
alpha =0.001


def run():
    true_val = X[(y==1).reshape(100),:]
    false_val = X[(y==0).reshape(100),:]
    plt.figure(figsize=(16,8))
    plt.scatter(true_val[:,1],true_val[:,2],marker='+',color='g')
    plt.scatter(false_val[:,1],false_val[:,2],marker='o',color='r')
    plt.xlabel('Exam 1 scores')
    plt.ylabel('Exam 2 scores')
    plt.legend(('Admitted', 'Not admitted'), bbox_to_anchor=(1.15, 1))
    plt.grid(color='gray', linestyle='--', linewidth=.6, axis='both', which='both', alpha=.4)
    plt.show()

    test_theta = np.array([-20, 0.2, 0.2])
    test_theta = op.minimize(fun=api.cost, x0=test_theta,args=(X,y), method='TNC',jac=api.gradient).x
    plot_x = np.array([X[:,1].min() - 2, X[:,1].max() + 2])
    plot_y = (-1./test_theta[2]) * (test_theta[1] * plot_x + test_theta[0])

    plt.figure(figsize=(16,8))
    plt.scatter(true_val[:,1],true_val[:,2],marker='+',color='g')
    plt.scatter(false_val[:,1],false_val[:,2],marker='o',color='r')
    plt.plot(plot_x,plot_y)
    plt.xlabel('Exam 1 scores')
    plt.ylabel('Exam 2 scores')
    plt.legend(('Admitted', 'Not admitted'), bbox_to_anchor=(1.15, 1))
    plt.grid(color='gray', linestyle='--', linewidth=.6, axis='both', which='both', alpha=.4)
    plt.show()

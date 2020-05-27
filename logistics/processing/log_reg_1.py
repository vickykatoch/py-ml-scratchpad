import numpy as np
import matplotlib.pyplot as plt
# from IPython.core.pylabtools import figsize
from os import path
import scipy.optimize as op

# sigmoid = 1/(1+e^-z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
   
def hypothesis(X, theta):
    return X @ theta

def cost(theta,X,y):
    m,n = X.shape
    hx = sigmoid(hypothesis(X,theta.reshape(len(theta),1)))
    J = np.sum((-y * np.log(hx)) - ((1-y) * np.log(1-hx))) * 1/m
    return J

def grad_desc(X,y,hx):
    m,n = X.shape
    return ((hx - y).T @ X) * 1/m


data = np.genfromtxt(path.join(path.dirname(__file__),'../data/raw/ex2data1.txt'),delimiter=",")
X = data[:,0:2]
y = data[:,2:]
m = len(X)
X = np.hstack([np.ones((m,1)),X])
m , n = X.shape
initial_theta = np.zeros((n,1))
alpha =0.001


def Gradient(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1))    
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m    
    return grad.flatten()

def grad_desc1(theta,X,y):
    hx = sigmoid(hypothesis(X,theta.reshape(len(theta),1)))
    g = ((hx - y).T @ X) * 1/m    
    return g.flatten()

def test():
    t_theta = np.array([-20, 0.2, 0.2])
    result = op.minimize(fun=cost, x0=t_theta,args=(X,y), method='TNC',jac=Gradient)
    print(result)

    t_theta = np.array([-20, 0.2, 0.2])
    result1 = op.minimize(fun=cost, x0=t_theta,args=(X,y), method='TNC',jac=grad_desc1)
    print(result1)

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

    theta = np.array([-25.16113549,   0.2062301 ,   0.20147003])
    plot_x = np.array([X[:,1].min() - 2, X[:,1].max() + 2])
    plot_y = (-1./theta[2]) * (theta[1] * plot_x + theta[0])

    plt.figure(figsize=(16,8))
    plt.scatter(true_val[:,1],true_val[:,2],marker='+',color='g')
    plt.scatter(false_val[:,1],false_val[:,2],marker='o',color='r')
    plt.plot(plot_x,plot_y)
    plt.xlabel('Exam 1 scores')
    plt.ylabel('Exam 2 scores')
    plt.legend(('Admitted', 'Not admitted'), bbox_to_anchor=(1.15, 1))
    plt.grid(color='gray', linestyle='--', linewidth=.6, axis='both', which='both', alpha=.4)
    plt.show()

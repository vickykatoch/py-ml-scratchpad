import numpy as np
import pandas as pd
import avam.common.file_helper as fh

#region FUNCTIONS
# Feature Normalization
def normalize(x):
    max_val = max(x)
    min_val = min(x)
    avg = np.mean(x)
    normalized = [(i-avg)/(max_val - min_val) for i in x]
    return normalized
# Hypothesis Function
def hypothesis_l(X,theta):
    n = len(theta)
    m = len(X)
    h = np.zeros((m,1))
    for i in range(0,m):
        sm=0
        for j in range(0,n):
            sm += X[i,j] * theta[j,0]
        h[i,0]=sm 
    return h

#Cost Function
def cost_l(h,y):
    m = len(h)
    cost = 0
    for i in range(m):
        cost += (h[i] - y[i]) ** 2
    return cost / (2*m)

#Gradient Descent Function
def gradient_descent_l(X,h,y, alpha,theta):
    n = X.shape[1]
    m = X.shape[0]
    new_thetas = np.zeros((n,1))
    for j in range(n):
        th = 0
        for i in range(m):
            th += (h[i,0] - y[i,0]) * X[i,j] 
        new_thetas[j,0] = theta[j,0] - (alpha * (1/m) * th)
    return new_thetas  
#endregion



def run():
    columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    df = pd.read_csv(fh.getHousingDataFile(), header=None, names=columns, sep='\s+')
    df = df.apply(normalize,axis=0)
    # Create X0 column
    Xd = df.drop(columns=['MEDV'])
    Xd.insert(0, 'X0', 1)
    # numpy array format
    X = Xd.values
    m = len(Xd.index)
    y = df.MEDV.to_numpy().reshape(m,1)
    n = Xd.shape[1]
    alpha = 0.0005
    iterations = 100000
    print('Sample Size : {}, Features : {}'.format(m,n))
    #Initialize Theta
    theta = np.ones((n,1))
    n_thetas = theta
    cost = np.zeros((iterations,1))
    for i in range(iterations):
        hypthesis = hypothesis_l(X,n_thetas)
        cost[i,0] = cost_l(hypthesis,y)
        n_thetas = gradient_descent_l(X,hypthesis,y, alpha,n_thetas)
        # if(i>0 and round(i/100,4)==1.0000):
        print("Iterations : {}".format(i))
    print(n_thetas)
    

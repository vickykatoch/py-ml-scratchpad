import numpy as np
import pandas as pd
import avam.common.file_helper as fh

#region Feature Normalization
# Feature Normalization
def normalize(x):
    max_val = max(x)
    min_val = min(x)
    avg = np.mean(x)
    normalized = [(i-avg)/(max_val - min_val) for i in x]
    return normalized
#endregion

#region HYPOTHESIS
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

def hypothesis_v(X,theta):
    return X @ theta
#enregion

#region COST
def cost_l(h,y):
    m = len(h)
    cost = np.zeros((m,1))
    for i in range(m):
        cost[i] = (h[i] - y[i]) ** 2
    return cost / (2*m)
#endregion

def run():
    columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    df = pd.read_csv(fh.getHousingDataFile(), header=None, names=columns, sep='\s+')
    Xd = df.drop(columns=['MEDV'])
    Xd.insert(0,'X0',1)
    m=len(Xd)
    n = Xd.shape[1]
    X = Xd.values
    y = df.MEDV.values.reshape(m,1)

    # Initialize theta
    theta = np.ones((n,1))

    h_l = hypothesis_l(X,theta)
    h_v = hypothesis_v(X,theta)
    output = np.hstack([h_l,h_v])
    print(output)

    df1 = df.copy()
    df1.apply(normalize,axis=0)

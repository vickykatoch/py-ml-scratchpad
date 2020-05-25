
import numpy as np


def getHousingDataColumnNames():
    return ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# Feature Normalization
def normalize(x):
    max_val = max(x)
    min_val = min(x)
    avg = np.mean(x)
    normalized = [(i-avg)/(max_val - min_val) for i in x]
    return normalized

def getInitialData(df):
    df = df.apply(normalize,axis=0)
    # Create X0 column
    Xd = df.drop(columns=['MEDV'])
    X = Xd.values
    m = len(Xd.index)
    n = Xd.shape[1]
    y = df.MEDV.to_numpy().reshape(m,1)
    
    return (df, X, y, m, n)

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

def gradient_descent_l(X,h,y, alpha,theta):
    initial_theta = theta.copy()
    (m,n)= X.shape
    for i in range(m):
        pa_del = h[i,0] - y[i,0]
        for k in range(n):
            initial_theta[k,0] = initial_theta[k,0] - (alpha * (1/m) * (pa_del * X[i,k]))
    return initial_theta

# Hypothesis Function
def hypothesis_v(X,theta):
    return X@theta

#Cost Function
def cost_v(h,y):
    m = len(h)
    return np.sum((h - y) ** 2) * (1/(2*m))

def gradient_descent_v(X,h,y, alpha,theta):
    m = X.shape[0]
    return theta - alpha * (1/m) * np.transpose(X)@(X@theta - y)
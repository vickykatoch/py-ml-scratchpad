import numpy as np
import pandas as pd
import regression.helpers.data_helper as data_helper
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def visualize(X,y):
    plt.scatter(X, y, color='blue')  # you can use test_data_X and test_data_Y instead.
    # plt.plot([min_x, max_x], [b, m*max_x + b], 'r')
    plt.title('Fitted linear regression', fontsize=16)
    plt.xlabel('x', fontsize=13)
    plt.ylabel('y', fontsize=13)
    plt.show()

def run():
    df = pd.read_csv(data_helper.getCaliforniaHousingCsv())
    # Reduce the Data size by 75%
    df = df.sample(frac=0.25)
    m = len(df)
    # Collect our input features and labels
    X = df['median_income'].to_numpy().reshape(m,1)
    y = df['median_house_value'].to_numpy().reshape(m,1)
    model = LinearRegression()
    model.fit(X,y)
    print("Y intercept : {}, Slope (y=mx+b) : {}".format(model.intercept_, model.coef_))
    predictions = model.predict(X)
    visualize(X,y)
    
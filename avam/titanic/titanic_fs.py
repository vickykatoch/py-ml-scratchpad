import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os import path
import pickle
import sklearn

print(sklearn.__version__)
data_dir = path.join(path.dirname(__file__), '../../data/titanic')


def print_scores(title, model, X, y):
    print('******************************************************')
    print(f'{title}')
    print('******************************************************')
    print('Score : {0:.5f}'.format(model.score(X, y)))
    print('Accuracy : {0:.5f}'.format(accuracy_score(y, model.predict(X))))
    print('Confusion matrix for base line model is : \n {}'.format(
        confusion_matrix(y, model.predict(X))))
    print('Precision score for base line model is : {0:.2f}'.format(
        precision_score(y, model.predict(X))))
    print('Recall score for base line model is : {0:.2f}'.format(
        recall_score(y, model.predict(X))))
    print('******************************************************\n\n')


# import sklearn
def run():
    # Load Processed data
    train_df = pd.read_csv(
        path.join(data_dir, 'processed/train.csv'), index_col="PassengerId")
    test_df = pd.read_csv(
        path.join(data_dir, 'processed/test.csv'), index_col='PassengerId')

    # Split Features and output
    X = np.matrix(train_df.loc[:, 'Age':].astype('float'))
    y = train_df.Survived.ravel()

    # Split Train dataset into train and test in 8:2 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0)

    model_log_reg = LogisticRegression(random_state=0, max_iter=10000)
    model_log_reg.fit(X_train, y_train)

    print_scores('1. LogisticRegression - Training Dataset',
                 model_log_reg, X_test, y_test)

    parameters = {'C': [1.0, 10.0, 50.0, 100.0, 1000.0],
                  'penalty': ['l1', 'l2']}
    clf = GridSearchCV(model_log_reg, param_grid=parameters, cv=3)
    clf.fit(X_train, y_train)

    print(f'Best Scrore : {clf.best_params_}')
    Xt = np.matrix(test_df.loc[:, 'Age':].astype('float'))
    predictions = model_log_reg.predict(Xt)
    sub_df = pd.DataFrame(
        {'PassengerId': test_df.index, 'Survived': predictions})
    print(f'Survived : {len(sub_df[sub_df.Survived==1])}')
    sub_df.to_csv(path.join(data_dir, 'external/03_dummy.csv'), index=False)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(
        f'Min : {X_train_scaled[:,0].min()}, Max : {X_train_scaled[:,0].min()}')

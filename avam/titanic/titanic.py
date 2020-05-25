import pandas as pd
import numpy as np
from os import path
import avam.titanic.utils as tu



def read_data():    
    raw_data_path = path.join(path.dirname(__file__), path.pardir,path.pardir,'data','titanic','raw')
    train_file_path = path.join(raw_data_path, 'titanic-train.csv')
    test_file_path = path.join(raw_data_path, 'titanic-test.csv')
    train_df = pd.read_csv(train_file_path,index_col="PassengerId")
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')
    test_df['Survived'] = -888
    return pd.concat((train_df, test_df), axis=0, sort=True)

def process_data(df):
    return (
        df.assign(Title = lambda x: x.Name.map(tu.get_title))
         # working missing values - start with this
         .pipe(tu.fill_missing_values)
         # create fare bin feature
         .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low','low','high','very_high']))
         # create age state
         .assign(AgeState = lambda x : np.where(x.Age >= 18, 'Adult','Child'))
         .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)
         .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))
          # create deck feature
         .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin)) 
         .assign(Deck = lambda x : x.Cabin.map(tu.get_deck))
         # feature encoding 
         .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1,0))
         .pipe(pd.get_dummies, columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])
         # add code to drop unnecessary columns
         .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1)
         # reorder columns
         .pipe(tu.reorder_columns)
    )

def write_data(df):
    processed_data_path = path.join(path.dirname(__file__), path.pardir,path.pardir,'data','titanic','processed')
    write_train_path = path.join(processed_data_path, 'train.csv')
    write_test_path = path.join(processed_data_path, 'test.csv')
    # train data
    df[df.Survived != -888].to_csv(write_train_path) 
    # test data
    columns = [column for column in df.columns if column != 'Survived']
    df[df.Survived == -888][columns].to_csv(write_test_path) 

def run():
    df = read_data()
    df = process_data(df)
    write_data(df)
# TITANIC DATA PROCESSOR

import pandas as pd
import numpy as np
# from os import path


def get_title(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def fill_missing_values(df):
  # embarked
  df.Embarked.fillna('C', inplace=True)
  # fare
  median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
  df.Fare.fillna(median_fare, inplace=True)
  # age
  title_age_median = df.groupby('Title').Age.transform('median')
  df.Age.fillna(title_age_median , inplace=True)
  return df

def get_deck(cabin):
  return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def reorder_columns(df):
  columns = [column for column in df.columns if column != 'Survived']
  columns = ['Survived'] + columns
  df = df[columns]
  return df
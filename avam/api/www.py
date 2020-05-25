from flask import Flask, request
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

columns = ['Age', 'Fare', 'FamilySize', 'IsMother', 'IsMale', 'Deck_A',
           'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_Z',
           'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Lady', 'Title_Master',
           'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Sir',
           'Fare_Bin_very_low', 'Fare_Bin_low', 'Fare_Bin_high',
           'Fare_Bin_very_high', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
           'AgeState_Adult', 'AgeState_Child']


@app.route('/api', methods=['POST'])
def sayHello():
    data = request.get_json(force=True)
    name = data['name']
    return f"Hello {name}"


@app.route('/predict', methods=['POST'])
def predict():
    data = json.dumps(request.get_json(force=True))
    df = pd.read_json(data)
    px_ids = df['PassengerId'].ravel()
    actuals = df['Survived'].ravel()

    X = np.matrix(df[columns].astype('float'))

    name = data['name']
    return f"Hello {name}"


def run():
    app.run(port=10000, debug=True)
    print(f"Web server listening on port : 10000")


if __name__ == '__main__':
    run()

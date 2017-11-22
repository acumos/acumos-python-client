# -*- coding: utf-8 -*-
'''
Generates a model and dumps it to present working directory for testing purposes
'''
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from acumos.modeling import Model, List, create_namedtuple, create_dataframe
from acumos.session import AcumosSession


if __name__ == '__main__':
    '''Main'''

    iris = load_iris()
    X = iris.data
    y = iris.target

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    X_df = pd.DataFrame(X, columns=columns)

    DataFrame = create_dataframe('DataFrame', X_df)
    Predictions = create_namedtuple('Predictions', [('predictions', List[int])])

    def predict(df: DataFrame) -> Predictions:
        '''Predicts the class of iris'''
        X = np.column_stack(df)
        yhat = clf.predict(X)
        preds = Predictions(predictions=yhat)
        return preds

    model = Model(transform=predict)

    s = AcumosSession(None)
    s.dump(model, 'model', '.')

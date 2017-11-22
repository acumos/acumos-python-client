# -*- coding: utf-8 -*-
"""
Provides a scikit-learn Acumos model example
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from acumos.modeling import Model, List, create_dataframe
from acumos.session import AcumosSession


iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X_df = pd.DataFrame(X, columns=columns)

IrisDataFrame = create_dataframe('IrisDataFrame', X_df)


def classify_iris(df: IrisDataFrame) -> List[int]:
    '''Returns an array of iris classifications'''
    X = np.column_stack(df)
    return clf.predict(X)


model = Model(classify=classify_iris)

session = AcumosSession()
session.dump(model, 'my-iris', '.')  # creates ./my-iris

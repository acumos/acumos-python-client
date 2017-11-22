# -*- coding: utf-8 -*-
"""
Provides a keras Acumos model example
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense

from acumos.session import AcumosSession
from acumos.modeling import Model, List, create_dataframe


iris = load_iris()
X = iris.data
y = pd.get_dummies(iris.target).values

clf = Sequential()
clf.add(Dense(3, input_dim=4, activation='relu'))
clf.add(Dense(3, activation='softmax'))
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.fit(X, y)

X_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
IrisDataFrame = create_dataframe('IrisDataFrame', X_df)


def classify_iris(df: IrisDataFrame) -> List[int]:
    '''Returns an array of iris classifications'''
    X = np.column_stack(df)
    return clf.predict(X)


model = Model(classify=classify_iris)

session = AcumosSession()
session.dump(model, 'model', '.')  # creates ./model

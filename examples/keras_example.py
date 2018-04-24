# -*- coding: utf-8 -*-
# ===============LICENSE_START=======================================================
# Acumos Apache-2.0
# ===================================================================================
# Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
# ===================================================================================
# This Acumos software file is distributed by AT&T and Tech Mahindra
# under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============LICENSE_END=========================================================
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
    return clf.predict_classes(X)


model = Model(classify=classify_iris)

session = AcumosSession()
session.dump(model, 'model', '.')  # creates ./model

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
Provides a scikit-learn Acumos model example
"""
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from acumos.modeling import Model, List, create_namedtuple
from acumos.session import AcumosSession


iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

IrisDataFrame = create_namedtuple('IrisDataFrame', [('sepal_length', List[float]),
                                                    ('sepal_width', List[float]),
                                                    ('petal_length', List[float]),
                                                    ('petal_width', List[float])])

# =============================================================================
# # starting in Python 3.6, you can alternatively use this simpler syntax:
#
# from acumos.modeling import NamedTuple
#
# class IrisDataFrame(NamedTuple):
#     '''DataFrame corresponding to the Iris dataset'''
#     sepal_length: List[float]
#     sepal_width: List[float]
#     petal_length: List[float]
#     petal_width: List[float]
# =============================================================================

# =============================================================================
# # A pandas DataFrame can also be used to infer appropriate NamedTuple types:
#
# import pandas as pd
# from acumos.modeling import create_dataframe
#
# X_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# IrisDataFrame = create_dataframe('IrisDataFrame', X_df)
# =============================================================================


def classify_iris(df: IrisDataFrame) -> List[int]:
    '''Returns an array of iris classifications'''
    X = np.column_stack(df)
    return clf.predict(X)


model = Model(classify=classify_iris)

session = AcumosSession()
session.dump(model, 'my-iris', '.')  # creates ./my-iris

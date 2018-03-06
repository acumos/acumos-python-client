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

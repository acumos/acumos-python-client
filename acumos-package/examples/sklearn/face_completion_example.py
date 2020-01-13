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

Example adapted from http://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html
"""
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.neighbors import KNeighborsRegressor

from acumos.modeling import Model, List, create_namedtuple
from acumos.session import AcumosSession


# =============================================================================
# from the original example above
# =============================================================================

# Load the faces datasets
data = fetch_olivetti_faces()
targets = data.target

data = data.images.reshape((len(data.images), -1))
train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

n_pixels = data.shape[1]
# Upper half of the faces
X_train = train[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

# =============================================================================
# Acumos specific code
# =============================================================================

# represents a single "flattened" [1 x n] image array
FlatImage = create_namedtuple('FlatImage', [('image', List[float])])

# represents a collection of flattened image arrays
FlatImages = create_namedtuple('FlatImages', [('images', List[FlatImage])])


def complete_faces(images: FlatImages) -> FlatImages:
    '''Predicts the bottom half of each input image'''
    X = np.vstack(images).squeeze()  # creates an [m x n] matrixs with m images and n pixels
    yhat = knn.predict(X)
    return FlatImages([FlatImage(row) for row in yhat])


model = Model(complete_faces=complete_faces)

session = AcumosSession()
session.dump(model, 'face-model', '.')  # creates ./face-model

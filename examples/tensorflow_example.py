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
Provides a TensorFlow Acumos model example
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

from acumos.session import AcumosSession
from acumos.modeling import Model, List, create_dataframe


iris = load_iris()
data = iris.data
target = iris.target
target_onehot = pd.get_dummies(target).values.astype(float)

# =============================================================================
# build model
# =============================================================================

x = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y = tf.placeholder(shape=[None, 3], dtype=tf.float32)

layer1 = tf.layers.dense(x, 3, activation=tf.nn.relu)
layer2 = tf.layers.dense(layer1, 3, activation=tf.nn.relu)
logits = tf.layers.dense(layer2, 3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(0.075).minimize(cost)

# =============================================================================
# train model & predict
# =============================================================================

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run([init])

for epoch in range(1000):
    _, loss = sess.run([optimizer, cost], feed_dict={x: data, y: target_onehot})
    print("Epoch {} | Loss {}".format(epoch, loss))

prediction = tf.argmax(logits, 1)
yhat = sess.run([prediction], {x: data})[0]

# note: this predicts on the training set for illustration purposes only
print(classification_report(target, yhat))

# =============================================================================
# create a acumos model from the tensorflow model
# =============================================================================

X_df = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
IrisDataFrame = create_dataframe('IrisDataFrame', X_df)


def classify_iris(df: IrisDataFrame) -> List[int]:
    '''Returns an array of iris classifications'''
    X = np.column_stack(df)
    return prediction.eval({x: X}, sess)


model = Model(classify=classify_iris)

session = AcumosSession()
session.dump(model, 'model', '.')  # creates ./model

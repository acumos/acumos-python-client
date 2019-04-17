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
Tests custom pickling logic
"""
import os
import sys
from tempfile import TemporaryDirectory
from os.path import join as path_join

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras_contrib.layers import PELU

from acumos.pickler import dump_model, load_model, AcumosContextManager, get_context
from acumos.exc import AcumosError
from acumos.modeling import Model

from utils import run_command, TEST_DIR
from user_module import user_function


_UNPICKLER_HELPER = path_join(TEST_DIR, 'unpickler_helper.py')


def test_user_script():
    '''Tests that user scripts are identified as dependencies'''

    def predict(x: int) -> int:
        return user_function(x)

    model = Model(predict=predict)

    with AcumosContextManager() as context:
        model_path = context.build_path('model.pkl')
        with open(model_path, 'wb') as f:
            dump_model(model, f)

            assert 'user_module' in context.script_names

        # unpickling should fail because `user_module` is not available
        with pytest.raises(Exception, match="No module named 'user_module'"):
            run_command([sys.executable, _UNPICKLER_HELPER, context.abspath])


def test_keras_contrib():
    '''Tests keras_contrib layer is saved correctly'''
    model = Sequential()
    model.add(Dense(10, input_shape=(10,)))
    model.add(PELU())

    model.compile(loss='mse', optimizer='adam')
    model.fit(x=np.random.random((10, 10)), y=np.random.random((10, 10)), epochs=1, verbose=0)

    with AcumosContextManager() as context:
        model_path = context.build_path('model.pkl')
        with open(model_path, 'wb') as f:
            dump_model(model, f)
            assert {'keras', 'dill', 'acumos', 'h5py', 'tensorflow', 'keras_contrib'} == context.package_names

        # verify that the contrib layers don't cause a load error
        run_command([sys.executable, _UNPICKLER_HELPER, context.abspath])


def test_function_import():
    '''Tests that a module used by a function is captured correctly'''
    import numpy as np

    def foo():
        return np.arange(5)

    with AcumosContextManager() as context:
        model_path = context.build_path('model.pkl')
        with open(model_path, 'wb') as f:
            dump_model(foo, f)

        assert {'dill', 'acumos', 'numpy'} == context.package_names

        with open(model_path, 'rb') as f:
            loaded_model = load_model(f)

    assert (loaded_model() == np.arange(5)).all()


def test_pickler_keras():
    '''Tests keras dump / load functionality'''
    iris = load_iris()
    X = iris.data
    y_onehot = pd.get_dummies(iris.target).values

    # test both keras and tensorflow.keras packages
    keras_pkg_names = {'keras', 'dill', 'acumos', 'h5py', 'tensorflow'}
    tf_pkg_names = {'dill', 'acumos', 'h5py', 'tensorflow'}

    for seq_cls, dense_cls, pkg_names in ((Sequential, Dense, keras_pkg_names),
                                          (tf.keras.Sequential, tf.keras.layers.Dense, tf_pkg_names)):
        model = seq_cls()
        model.add(dense_cls(3, input_dim=4, activation='relu'))
        model.add(dense_cls(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y_onehot, verbose=0)

        with TemporaryDirectory() as root:

            with AcumosContextManager(root) as context:
                model_path = context.build_path('model.pkl')
                with open(model_path, 'wb') as f:
                    dump_model(model, f)

                assert pkg_names == context.package_names

            with AcumosContextManager(root) as context:
                with open(model_path, 'rb') as f:
                    loaded_model = load_model(f)

        assert (model.predict_classes(X, verbose=0) == loaded_model.predict_classes(X, verbose=0)).all()


def test_pickler_sklearn():
    '''Tests sklearn dump / load functionality'''
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = SVC()
    model.fit(X, y)

    with TemporaryDirectory() as root:

        with AcumosContextManager(root) as context:
            model_path = context.build_path('model.pkl')
            with open(model_path, 'wb') as f:
                dump_model(model, f)

            assert {'sklearn', 'dill', 'acumos', 'numpy'} == context.package_names

        with AcumosContextManager(root) as context:
            with open(model_path, 'rb') as f:
                loaded_model = load_model(f)

    assert (model.predict(X) == loaded_model.predict(X)).all()


def test_pickler_tensorflow():
    '''Tests tensorflow session and graph serialization'''
    tf.set_random_seed(0)

    iris = load_iris()
    data = iris.data
    target = iris.target
    target_onehot = pd.get_dummies(target).values.astype(float)

    with tf.Graph().as_default():

        # test pickling a session with trained weights

        session = tf.Session()
        x, y, prediction = _build_tf_model(session, data, target_onehot)
        yhat = session.run([prediction], {x: data})[0]

        with TemporaryDirectory() as model_root:
            with AcumosContextManager(model_root) as context:
                model_path = context.build_path('model.pkl')
                with open(model_path, 'wb') as f:
                    dump_model(session, f)

                assert {'acumos', 'dill', 'tensorflow'} == context.package_names

            with AcumosContextManager(model_root) as context:
                with open(model_path, 'rb') as f:
                    loaded_session = load_model(f)

            loaded_graph = loaded_session.graph
            loaded_prediction = loaded_graph.get_tensor_by_name(prediction.name)
            loaded_x = loaded_graph.get_tensor_by_name(x.name)
            loaded_yhat = loaded_session.run([loaded_prediction], {loaded_x: data})[0]

            assert loaded_session is not session
            assert loaded_graph is not session.graph
            assert (yhat == loaded_yhat).all()

        # tests pickling a session with a frozen graph

        with TemporaryDirectory() as frozen_root:
            save_path = path_join(frozen_root, 'model')

            with loaded_session.graph.as_default():
                saver = tf.train.Saver()
                saver.save(loaded_session, save_path)

            frozen_path = _freeze_graph(frozen_root, ['prediction'])
            frozen_graph = _unfreeze_graph(frozen_path)
            frozen_session = tf.Session(graph=frozen_graph)

        with TemporaryDirectory() as model_root:
            with AcumosContextManager(model_root) as context:
                model_path = context.build_path('model.pkl')
                with open(model_path, 'wb') as f:
                    dump_model(frozen_session, f)

            with AcumosContextManager(model_root) as context:
                with open(model_path, 'rb') as f:
                    loaded_frozen_session = load_model(f)

            loaded_frozen_graph = loaded_frozen_session.graph
            loaded_frozen_prediction = loaded_frozen_graph.get_tensor_by_name(prediction.name)
            loaded_frozen_x = loaded_frozen_graph.get_tensor_by_name(x.name)
            loaded_frozen_yhat = loaded_frozen_session.run([loaded_frozen_prediction], {loaded_frozen_x: data})[0]

            assert loaded_frozen_session is not frozen_session
            assert loaded_frozen_graph is not frozen_session.graph
            assert (yhat == loaded_frozen_yhat).all()


def _build_tf_model(session, data, target):
    '''Builds and iris tensorflow model and returns the prediction tensor'''
    x = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    y = tf.placeholder(shape=[None, 3], dtype=tf.float32)

    layer1 = tf.layers.dense(x, 3, activation=tf.nn.relu)
    logits = tf.layers.dense(layer1, 3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(0.075).minimize(cost)

    init = tf.global_variables_initializer()
    session.run(init)

    for epoch in range(3):
        _, loss = session.run([optimizer, cost], feed_dict={x: data, y: target})

    prediction = tf.argmax(logits, 1, name='prediction')
    return x, y, prediction


def _freeze_graph(model_dir, output_node_names):
    '''Modified from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc'''
    input_checkpoint = tf.train.get_checkpoint_state(model_dir).model_checkpoint_path
    graph_path = "{}.meta".format(input_checkpoint)
    output_graph = path_join(model_dir, 'frozen_model.pb')

    with tf.Session(graph=tf.Graph()) as session:
        saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
        saver.restore(session, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(session,
                                                                        session.graph.as_graph_def(),
                                                                        output_node_names)
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    return output_graph


def _unfreeze_graph(frozen_graph_path):
    '''Modified from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc'''
    with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def test_nested_model():
    '''Tests nested models'''
    iris = load_iris()
    X = iris.data
    y = iris.target
    y_onehot = pd.get_dummies(iris.target).values

    m1 = Sequential()
    m1.add(Dense(3, input_dim=4, activation='relu'))
    m1.add(Dense(3, activation='softmax'))
    m1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    m1.fit(X, y_onehot, verbose=0)

    m2 = SVC()
    m2.fit(X, y)

    # lambda watch out
    crazy_good_model = lambda x: m1.predict_classes(x) + m2.predict(x)  # noqa
    out1 = crazy_good_model(X)

    with TemporaryDirectory() as root:

        with AcumosContextManager(root) as context:
            model_path = context.build_path('model.pkl')
            with open(model_path, 'wb') as f:
                dump_model(crazy_good_model, f)

            assert {'sklearn', 'keras', 'dill', 'acumos', 'numpy', 'h5py', 'tensorflow'} == context.package_names

        with AcumosContextManager(root) as context:
            with open(model_path, 'rb') as f:
                loaded_model = load_model(f)

    out2 = loaded_model(X)
    assert (out1 == out2).all()


def test_context():
    '''Tests basic AcumosContextManager functionality'''
    with AcumosContextManager() as c1:
        c2 = get_context()
        assert c1 is c2
        assert {'dill', 'acumos'} == c1.package_names

        # default context already exists
        with pytest.raises(AcumosError):
            with AcumosContextManager():
                pass

        assert os.path.isdir(c1.abspath)

        abspath = c1.create_subdir()
        assert os.path.isdir(abspath)

    # context removes a temporary directory it creates
    assert not os.path.isdir(c1.abspath)

    # default context doesn't exist outside of CM
    with pytest.raises(AcumosError):
        get_context()


def test_context_provided_root():
    '''Tests AcumosContextManager with a provided root directory'''
    with TemporaryDirectory() as root:
        with AcumosContextManager(root) as c1:
            abspath = c1.create_subdir()
            assert os.path.isdir(abspath)

        # context does not remove a provided directory
        assert os.path.isdir(c1.abspath)
        assert os.path.isdir(abspath)


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

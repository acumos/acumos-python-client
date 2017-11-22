# -*- coding: utf-8 -*-
"""
Tests custom pickling logic
"""
import os
import tempfile

import pytest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense

from acumos.pickler import dump_model, load_model, AcumosContextManager, get_context
from acumos.exc import AcumosError


def test_function_import():
    '''Tests that a module used by a function is captured correctly'''
    import numpy as np

    def foo():
        return np.arange(5)

    with AcumosContextManager() as context:
        model_path = context.build_path('model.pkl')
        with open(model_path, 'wb') as f:
            dump_model(foo, f)

        assert {'dill', 'acumos', 'numpy'} == context.modules

        with open(model_path, 'rb') as f:
            loaded_model = load_model(f)

    assert (loaded_model() == np.arange(5)).all()


def test_pickler_keras():
    '''Tests keras dump / load functionality'''
    iris = load_iris()
    X = iris.data
    y_onehot = pd.get_dummies(iris.target).values

    model = Sequential()
    model.add(Dense(3, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y_onehot, verbose=0)

    with tempfile.TemporaryDirectory() as root:

        with AcumosContextManager(root) as context:
            model_path = context.build_path('model.pkl')
            with open(model_path, 'wb') as f:
                dump_model(model, f)

            assert {'keras', 'dill', 'acumos', 'h5py'} == context.modules

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

    with tempfile.TemporaryDirectory() as root:

        with AcumosContextManager(root) as context:
            model_path = context.build_path('model.pkl')
            with open(model_path, 'wb') as f:
                dump_model(model, f)

            assert {'sklearn', 'dill', 'acumos', 'numpy'} == context.modules

        with AcumosContextManager(root) as context:
            with open(model_path, 'rb') as f:
                loaded_model = load_model(f)

    assert (model.predict(X) == loaded_model.predict(X)).all()


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

    with tempfile.TemporaryDirectory() as root:

        with AcumosContextManager(root) as context:
            model_path = context.build_path('model.pkl')
            with open(model_path, 'wb') as f:
                dump_model(crazy_good_model, f)

            assert {'sklearn', 'keras', 'dill', 'acumos', 'numpy', 'h5py'} == context.modules

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
        assert {'dill', 'acumos'} == c1.modules

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
    with tempfile.TemporaryDirectory() as root:
        with AcumosContextManager(root) as c1:
            abspath = c1.create_subdir()
            assert os.path.isdir(abspath)

        # context does not remove a provided directory
        assert os.path.isdir(c1.abspath)
        assert os.path.isdir(abspath)


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

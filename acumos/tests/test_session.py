# -*- coding: utf-8 -*-
"""
Provides session tests
"""
import contextlib
import mock
import tempfile
import json
from os import environ, listdir
from os.path import isfile, dirname, abspath, join as path_join

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from jsonschema import validate

from acumos.wrapped import _infer_model_dir
from acumos.modeling import Model, List, create_namedtuple, create_dataframe
from acumos.session import AcumosSession, _dump_model, Requirements
from acumos.exc import AcumosError
from acumos.utils import load_artifact
from acumos.auth import clear_jwt, _USERNAME_VAR, _PASSWORD_VAR

from mock_server import MockServer, MODEL_URI, AUTH_URI


_TEST_DIR = dirname(abspath(__file__))
_REQ_FILES = ('model.zip', 'model.proto', 'metadata.json')
_CUSTOM_PACKAGE_DIR = path_join(_TEST_DIR, 'custom_package')
_FAKE_USERNAME = 'foo'
_FAKE_PASSWORD = 'bar'


with open(path_join(_TEST_DIR, 'schema-0.4.0.json')) as f:
    _SCHEMA = json.load(f)


def test_auth_envvar():
    '''Tests that environment variables can be used in place of interactive input'''
    clear_jwt()
    with _patch_environ(**{_USERNAME_VAR: _FAKE_USERNAME, _PASSWORD_VAR: _FAKE_PASSWORD}):
        _push_dummy_model()


def test_custom_package():
    '''Tests that custom packages can be included'''
    from custom_package import custom_module  # local to tests/ directory

    def my_transform(x: int, y: int) -> int:
        return custom_module.add_numbers(x, y)

    model = Model(transform=my_transform)
    model_name = 'my-model'

    reqs = Requirements(packages=[_CUSTOM_PACKAGE_DIR])

    s = AcumosSession()

    with tempfile.TemporaryDirectory() as tdir:

        with pytest.raises(AcumosError):
            s.dump(model, model_name, tdir)  # fails because `custom_package` is not declared

        s.dump(model, model_name, tdir, reqs)


def test_session_dump():
    '''Tests session dump'''

    def my_transform(x: int, y: int) -> int:
        return x + y

    model = Model(transform=my_transform)
    model_name = 'my-model'

    s = AcumosSession()

    with tempfile.TemporaryDirectory() as tdir:

        s.dump(model, model_name, tdir)
        model_dir = path_join(tdir, model_name)
        assert set(listdir(model_dir)) == set(_REQ_FILES)

        with pytest.raises(AcumosError):
            s.dump(model, model_name, tdir)  # file already exists


def test_dump_model():
    '''Tests dump model utility, including generated artifacts'''

    def func() -> None:
        pass

    model = Model(transform=func)
    model_name = 'my-model'

    reqs = Requirements(reqs=['wronglib'], req_map={'wronglib': 'scipy'}, packages=[_CUSTOM_PACKAGE_DIR])

    with _dump_model(model, model_name, reqs) as dump_dir:

        assert set(listdir(dump_dir)) == set(_REQ_FILES)

        metadata = load_artifact(dump_dir, 'metadata.json', module=json, mode='r')
        validate(metadata, _SCHEMA)

        # test that a user-provided library was included and correctly mapped
        assert 'scipy' in {r['name'] for r in metadata['runtime']['dependencies']['pip']['requirements']}

        # test that custom code was bundled
        model_dir = _infer_model_dir(dump_dir)
        assert isfile(path_join(model_dir, 'scripts', 'user_provided', 'custom_package', 'custom_module.py'))
        assert isfile(path_join(model_dir, 'scripts', 'user_provided', 'custom_package', '__init__.py'))


def test_session_push_sklearn():
    '''Tests basic model pushing functionality with sklearn'''
    clear_jwt()

    with _patch_auth():
        with MockServer():
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

            model = Model(predict=predict)

            s = AcumosSession(MODEL_URI, AUTH_URI)
            s.push(model, name='sklearn_iris_push')


def test_session_push_keras():
    '''Tests basic model pushing functionality with keras'''
    clear_jwt()

    with _patch_auth():
        with MockServer():
            iris = load_iris()
            X = iris.data
            y = pd.get_dummies(iris.target).values

            clf = Sequential()
            clf.add(Dense(3, input_dim=4, activation='relu'))
            clf.add(Dense(3, activation='softmax'))
            clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

            model = Model(predict=predict)

            s = AcumosSession(MODEL_URI, AUTH_URI)
            s.push(model, name='keras_iris_push')


def _push_dummy_model():
    '''Generic dummy model push routine'''

    def my_transform(x: int, y: int) -> int:
        return x + y

    model = Model(transform=my_transform)

    with MockServer():
        s = AcumosSession(MODEL_URI, AUTH_URI)
        s.push(model, name='my-model')


@contextlib.contextmanager
def _patch_auth():
    '''Convenience CM to patch session and auth modules for automated testing'''
    with mock.patch('acumos.auth.getuser', lambda x: _FAKE_USERNAME):
        with mock.patch('acumos.auth.getpass', lambda x: _FAKE_PASSWORD):
            with mock.patch('acumos.session._assert_valid_apis', _mock_assert_valid_apis):
                yield


@contextlib.contextmanager
def _patch_environ(**kwargs):
    '''Temporarily adds kwargs to os.environ'''
    try:
        environ.update(kwargs)
        yield
    finally:
        for k in kwargs.keys():
            del environ[k]


def _mock_assert_valid_apis(**kwargs):
    '''Mock _assert_valid_apis function that doesn't raise'''


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

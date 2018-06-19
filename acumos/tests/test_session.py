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
Provides session tests
"""
import contextlib
import mock
import tempfile
import json
import sys
from os import environ, listdir
from os.path import isfile, join as path_join

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
from acumos.session import AcumosSession, _dump_model
from acumos.exc import AcumosError
from acumos.utils import load_artifact
from acumos.auth import clear_jwt, _USERNAME_VAR, _PASSWORD_VAR, _TOKEN_VAR
from acumos.metadata import SCHEMA_VERSION, Requirements

from mock_server import MockServer
from utils import run_command, TEST_DIR
from user_package import user_package_module
from user_module import user_function


_USER_PACKAGE_DIR = path_join(TEST_DIR, 'user_package')
_MODEL_LOADER_HELPER = path_join(TEST_DIR, 'model_loader_helper.py')
_REQ_FILES = ('model.zip', 'model.proto', 'metadata.json')
_FAKE_USERNAME = 'foo'
_FAKE_PASSWORD = 'bar'
_FAKE_TOKEN = 'secrettoken'


def test_auth_envvar():
    '''Tests that environment variables can be used in place of interactive input'''
    clear_jwt()
    with _patch_environ(**{_USERNAME_VAR: _FAKE_USERNAME, _PASSWORD_VAR: _FAKE_PASSWORD}):
        _push_dummy_model()

    clear_jwt()
    with _patch_environ(**{_TOKEN_VAR: _FAKE_TOKEN}):
        _push_dummy_model()


def test_extra_header():
    '''Tests that extra headers are correctly sent to the onboarding server'''
    clear_jwt()
    # in the mock onboarding server, this extra test header acts as another auth header
    with _patch_environ(**{_TOKEN_VAR: 'wrongtoken'}):
        extra_headers = {'X-Test-Header': _FAKE_TOKEN}
        _push_dummy_model(extra_headers)

    clear_jwt()
    with pytest.raises(AcumosError):
        with _patch_environ(**{_TOKEN_VAR: 'wrongtoken'}):
            extra_headers = {'X-Test-Header': 'wrongtoken'}
            _push_dummy_model(extra_headers)


def test_custom_package():
    '''Tests that custom packages can be included, wrapped, and loaded'''

    def my_transform(x: int, y: int) -> int:
        return user_package_module.add_numbers(x, y)

    model = Model(transform=my_transform)
    model_name = 'my-model'

    # load should fail without requirements
    with pytest.raises(Exception, match='Module user_package was detected as a dependency'):
        with _dump_model(model, model_name) as dump_dir:
            pass

    reqs = Requirements(packages=[_USER_PACKAGE_DIR])

    with _dump_model(model, model_name, reqs) as dump_dir:
        run_command([sys.executable, _MODEL_LOADER_HELPER, dump_dir, 'user_package'])


def test_custom_script():
    '''Tests that custom modules can be included, wrapped, and loaded'''

    def predict(x: int) -> int:
        return user_function(x)

    model = Model(predict=predict)
    model_name = 'my-model'

    with _dump_model(model, model_name) as dump_dir:
        run_command([sys.executable, _MODEL_LOADER_HELPER, dump_dir, 'user_module'])


def test_script_req():
    '''Tests that Python scripts can be included using Requirements'''

    def predict(x: int) -> int:
        return x

    model = Model(predict=predict)
    model_name = 'my-model'

    # tests that individual script and directory of scripts are both gathered
    reqs = Requirements(scripts=_abspath('./user_module.py', './user_package'))

    with _dump_model(model, model_name, reqs) as dump_dir:
        _verify_files(dump_dir, ('scripts/user_provided/user_package_module.py',
                                 'scripts/user_provided/__init__.py',
                                 'scripts/user_provided/user_module.py'))

    bad_reqs = Requirements(scripts=_abspath('./user_module.py', './user_package', 'not_real.py'))

    with pytest.raises(AcumosError, match='does not exist'):
        with _dump_model(model, model_name, bad_reqs) as dump_dir:
            pass

    bad_reqs = Requirements(scripts=_abspath('./user_module.py', './user_package', './att.png'))

    with pytest.raises(AcumosError, match='is invalid'):
        with _dump_model(model, model_name, bad_reqs) as dump_dir:
            pass


def _abspath(*files):
    '''Returns the absolute path of a local test files'''
    return tuple(path_join(TEST_DIR, file) for file in files)


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

    def predict(x: int) -> int:
        return user_function(x)

    model = Model(predict=predict)
    model_name = 'my-model'

    reqs = Requirements(reqs=['wronglib'], req_map={'wronglib': 'scipy'}, packages=[_USER_PACKAGE_DIR])

    with _dump_model(model, model_name, reqs) as dump_dir:

        assert set(listdir(dump_dir)) == set(_REQ_FILES)

        metadata = load_artifact(dump_dir, 'metadata.json', module=json, mode='r')
        schema = _load_schema(SCHEMA_VERSION)
        validate(metadata, schema)

        # test that a user-provided library was included and correctly mapped
        assert 'scipy' in {r['name'] for r in metadata['runtime']['dependencies']['pip']['requirements']}

        # test that custom package was bundled
        _verify_files(dump_dir, ('scripts/user_provided/user_package/user_package_module.py',
                                 'scripts/user_provided/user_package/__init__.py',
                                 'scripts/user_provided/user_module.py'))


def _verify_files(dump_dir, files):
    '''Asserts that `files` exist in `dump_dir`'''
    model_dir = _infer_model_dir(dump_dir)
    for file in files:
        assert isfile(path_join(model_dir, file))


def _load_schema(version):
    '''Returns a jsonschema dict from the model-schema submodule'''
    path = path_join(TEST_DIR, 'schemas', "schema-{}.json".format(version))
    with open(path) as f:
        schema = json.load(f)
    return schema


def test_session_push_sklearn():
    '''Tests basic model pushing functionality with sklearn'''
    clear_jwt()

    with _patch_auth():
        with MockServer() as server:
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

            model_url, auth_url, _, _ = server.config
            s = AcumosSession(model_url, auth_url)
            s.push(model, name='sklearn_iris_push')


def test_session_push_keras():
    '''Tests basic model pushing functionality with keras'''
    clear_jwt()

    with _patch_auth():
        with MockServer() as server:
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

            model_url, auth_url, _, _ = server.config
            s = AcumosSession(model_url, auth_url)
            s.push(model, name='keras_iris_push')


def _push_dummy_model(extra_headers=None):
    '''Generic dummy model push routine'''

    def my_transform(x: int, y: int) -> int:
        return x + y

    model = Model(transform=my_transform)

    with MockServer() as server:
        model_url, auth_url, _, _ = server.config
        s = AcumosSession(model_url, auth_url)
        s.push(model, name='my-model', extra_headers=extra_headers)


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
        orig_vars = {k: environ[k] for k in kwargs.keys() if k in environ}
        environ.update(kwargs)
        yield
    finally:
        environ.update(orig_vars)
        for extra_key in (kwargs.keys() - orig_vars.keys()):
            del environ[extra_key]


def _mock_assert_valid_apis(**kwargs):
    '''Mock _assert_valid_apis function that doesn't raise'''


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

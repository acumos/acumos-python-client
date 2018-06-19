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
Provides wrapped model tests
"""
import io
import sys
import logging
from os.path import join as path_join
from collections import Counter
from operator import eq
from tempfile import TemporaryDirectory

import pytest
import PIL
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from google.protobuf.json_format import MessageToJson, MessageToDict

from acumos.wrapped import load_model, _pack_pb_msg
from acumos.modeling import Model, create_dataframe, List, Dict, create_namedtuple
from acumos.session import _dump_model, _copy_dir, Requirements

from test_pickler import _build_tf_model
from utils import TEST_DIR


_IMG_PATH = path_join(TEST_DIR, 'att.png')

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.mark.skipif(sys.version_info < (3, 6), reason='Requires python3.6')
def test_py36_namedtuple():
    '''Tests to make sure that new syntax for NamedTuple works with wrapping'''
    from py36_namedtuple import Input, Output

    def adder(data: Input) -> Output:
        return Output(data.x + data.y)

    _generic_test(adder, Input(1, 2), Output(3))


@pytest.mark.flaky(reruns=5)
def test_wrapped_prim_type():
    '''Tests model wrap and load functionality'''

    def f1(x: int, y: int) -> int:
        return x + y

    def f2(x: int, y: int) -> None:
        pass

    def f3() -> None:
        pass

    def f4() -> int:
        return 3330

    def f5(data: bytes) -> str:
        '''Something more complex'''
        buffer = io.BytesIO(data)
        img = PIL.Image.open(buffer)
        return img.format

    def f6(x: List[int]) -> int:
        return sum(x)

    def f7(x: List[str]) -> Dict[str, int]:
        return Counter(x)

    def f8(x: List[np.int32]) -> np.int32:
        return np.sum(x)

    # input / output "answers"
    f1_in = (1, 2)
    f1_out = (3, )

    f2_in = (1, 2)
    f2_out = ()

    f3_in = ()
    f3_out = ()

    f4_in = (0, )
    f4_out = (3330, )

    with open(_IMG_PATH, 'rb') as f:
        f5_in = (f.read(), )
        f5_out = ('PNG', )

    f6_in = ([1, 2, 3], )
    f6_out = (6, )

    f7_in = (['a', 'a', 'b'], )
    f7_out = ({'a': 2, 'b': 1}, )

    f8_in = ([1, 2, 3], )
    f8_out = (6, )

    for func, in_, out in ((f1, f1_in, f1_out), (f2, f2_in, f2_out), (f3, f3_in, f3_out),
                           (f4, f4_in, f4_out), (f6, f6_in, f6_out), (f8, f8_in, f8_out)):
        _generic_test(func, in_, out)

    _generic_test(f5, f5_in, f5_out, reqs=Requirements(req_map={'PIL': 'pillow'}))
    _generic_test(f7, f7_in, f7_out, skip=_dict_skips)


@pytest.mark.flaky(reruns=5)
def test_wrapped_nested_type():
    '''Tests to make sure that nested NamedTuple messages are unpacked correctly'''
    Inner = create_namedtuple('Inner', [('x', int), ('y', int), ('z', int)])

    N1 = create_namedtuple('N1', [('dict_data', Dict[str, int])])
    N2 = create_namedtuple('N2', [('n1s', List[N1])])

    def f1(x: List[Inner]) -> Inner:
        '''Returns the component-wise sum of a sequence of Inner'''
        sums = np.vstack(x).sum(axis=0)
        return Inner(*sums)

    def f2(n2_in: N2) -> N2:
        '''Returns another N2 type using data from the input N2 type'''
        n1_in = n2_in.n1s[0]
        dict_data = dict(**n1_in.dict_data)  # shallow copy
        dict_data['b'] = 2
        n1_out = N1(dict_data=dict_data)
        n2_out = N2(n1s=[n1_out, n1_out])
        return n2_out

    f1_in = ([Inner(1, 2, 3), ] * 5, )
    f1_out = (5, 10, 15)

    n1 = N1(dict_data={'a': 1})
    n1_out = N1(dict_data={'a': 1, 'b': 2})
    f2_in = N2(n1s=[n1])
    f2_out = N2(n1s=[n1_out, n1_out])

    _generic_test(f1, f1_in, f1_out)
    _generic_test(f2, f2_in, f2_out, skip=_dict_skips)


def _dict_skips(as_, from_):
    '''Skips byte and json str output comparison due to odd failures, perhaps related to dict ordering'''
    return as_ in {'as_pb_bytes', 'as_json'}


@pytest.mark.flaky(reruns=5)
def test_wrapped_sklearn():
    '''Tests model wrap and load functionality'''

    iris = load_iris()
    X = iris.data
    y = iris.target

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    yhat = clf.predict(X)

    columns = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    X_df = pd.DataFrame(X, columns=columns)
    IrisDataFrame = create_dataframe('IrisDataFrame', X_df)

    def f1(data: IrisDataFrame) -> List[int]:
        '''Creates a numpy ndarray and predicts'''
        X = np.column_stack(data)
        return clf.predict(X)

    def f2(data: IrisDataFrame) -> List[int]:
        '''Creates a pandas DataFrame and predicts'''
        X = np.column_stack(data)
        df = pd.DataFrame(X, columns=columns)
        return clf.predict(df.values)

    in_ = tuple(col for col in X.T)
    out = (yhat, )

    for func in (f1, f2):
        _generic_test(func, in_, out, wrapped_eq=lambda a, b: (a[0] == b[0]).all())


@pytest.mark.flaky(reruns=5)
def test_wrapped_tensorflow():
    '''Tests model wrap and load functionality'''
    tf.set_random_seed(0)

    iris = load_iris()
    data = iris.data
    target = iris.target
    target_onehot = pd.get_dummies(target).values.astype(float)

# =============================================================================
#     test with explicit session
# =============================================================================

    tf.reset_default_graph()

    session = tf.Session()
    x, y, prediction = _build_tf_model(session, data, target_onehot)
    yhat = session.run([prediction], {x: data})[0]

    X_df = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    IrisDataFrame = create_dataframe('IrisDataFrame', X_df)

    def f1(df: IrisDataFrame) -> List[int]:
        '''Tests with explicit session provided'''
        X = np.column_stack(df)
        return prediction.eval({x: X}, session)

    in_ = tuple(col for col in data.T)
    out = (yhat, )

    _generic_test(f1, in_, out, wrapped_eq=lambda a, b: (a[0] == b[0]).all(), preload=tf.reset_default_graph)

# =============================================================================
#     test with implicit default session
# =============================================================================

    tf.reset_default_graph()

    session = tf.InteractiveSession()
    x, y, prediction = _build_tf_model(session, data, target_onehot)
    yhat = session.run([prediction], {x: data})[0]

    def f2(df: IrisDataFrame) -> List[int]:
        '''Tests with implicit default session'''
        X = np.column_stack(df)
        return prediction.eval({x: X})

    in_ = tuple(col for col in data.T)
    out = (yhat, )

    _generic_test(f2, in_, out, wrapped_eq=lambda a, b: (a[0] == b[0]).all(), preload=tf.reset_default_graph)


def _generic_test(func, in_, out, wrapped_eq=eq, pb_mg_eq=eq, pb_bytes_eq=eq, dict_eq=eq, json_eq=eq, preload=None, reqs=None, skip=None):
    '''Reusable wrap test routine with swappable equality functions'''

    model = Model(transform=func)
    model_name = 'my-model'

    with TemporaryDirectory() as tdir:
        with _dump_model(model, model_name, reqs) as dump_dir:
            _copy_dir(dump_dir, tdir, model_name)

        if preload is not None:
            preload()

        copied_dump_dir = path_join(tdir, model_name)
        wrapped_model = load_model(copied_dump_dir)

        TransIn = model.transform.input_type
        TransOut = model.transform.output_type

        trans_in = TransIn(*in_)
        trans_out = TransOut(*out)

        trans_in_pb = _pack_pb_msg(trans_in, wrapped_model.transform._module)
        trans_out_pb = _pack_pb_msg(trans_out, wrapped_model.transform._module)

        trans_in_pb_bytes = trans_in_pb.SerializeToString()
        trans_out_pb_bytes = trans_out_pb.SerializeToString()

        trans_in_dict = MessageToDict(trans_in_pb)
        trans_out_dict = MessageToDict(trans_out_pb)

        trans_in_json = MessageToJson(trans_in_pb, indent=0)
        trans_out_json = MessageToJson(trans_out_pb, indent=0)

        # test all from / as combinations
        for as_method_name, as_data_expected, eq_func in (('as_wrapped', trans_out, wrapped_eq),
                                                          ('as_pb_msg', trans_out_pb, pb_mg_eq),
                                                          ('as_pb_bytes', trans_out_pb_bytes, pb_bytes_eq),
                                                          ('as_dict', trans_out_dict, dict_eq),
                                                          ('as_json', trans_out_json, json_eq)):
            for from_method_name, from_data in (('from_wrapped', trans_in),
                                                ('from_pb_msg', trans_in_pb),
                                                ('from_pb_bytes', trans_in_pb_bytes),
                                                ('from_dict', trans_in_dict),
                                                ('from_json', trans_in_json)):

                if skip is not None and skip(as_method_name, from_method_name):
                    logger.info("Skipping {} -> {}".format(from_method_name, as_method_name))
                    continue

                from_method = getattr(wrapped_model.transform, from_method_name)
                resp = from_method(from_data)
                as_data_method = getattr(resp, as_method_name)
                as_data = as_data_method()
                assert eq_func(as_data, as_data_expected)


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

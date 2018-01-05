# -*- coding: utf-8 -*-
"""
Provides wrapped model tests
"""
import tempfile
import io
import subprocess
import sys
import os
from os.path import join as path_join, abspath, dirname
from collections import Counter
from operator import eq

import pytest
import PIL
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from acumos.wrapped import load_model, _pack_pb_msg
from acumos.modeling import Model, create_dataframe, List, Dict, create_namedtuple
from acumos.session import _dump_model, _copy_dir, Requirements

from utils import get_workspace


_TEST_DIR = dirname(abspath(__file__))
_IMG_PATH = path_join(_TEST_DIR, 'att.png')
_CUSTOM_PACKAGE_DIR = path_join(_TEST_DIR, 'custom_package')
_CUSTOM_PACKAGE_HELPER = path_join(_TEST_DIR, 'custom_package_test_helper.py')


def test_custom_package():
    '''Tests that custom packages can be included, wrapped, and loaded'''
    from custom_package import custom_module  # local to tests/ directory

    def my_transform(x: int, y: int) -> int:
        return custom_module.add_numbers(x, y)

    model = Model(transform=my_transform)
    model_name = 'my-model'

    reqs = Requirements(packages=[_CUSTOM_PACKAGE_DIR])

    with _dump_model(model, model_name, reqs) as dump_dir:
        workspace = get_workspace()
        test = subprocess.Popen([sys.executable, _CUSTOM_PACKAGE_HELPER, dump_dir],
                                stderr=subprocess.PIPE, env={'PYTHONPATH': workspace,
                                                             'PATH': os.environ['PATH']})
        _, stderr = test.communicate()
        assert test.returncode == 0, stderr.decode()


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
        return 0

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

    f4_in = ()
    f4_out = (0, )

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
                           (f4, f4_in, f4_out), (f5, f5_in, f5_out), (f6, f6_in, f6_out),
                           (f7, f7_in, f7_out), (f8, f8_in, f8_out)):

        _generic_test(func, in_, out)


@pytest.mark.flaky(reruns=5)
def test_wrapped_nested_type():
    '''Tests to make sure that nested NamedTuple messages are unpacked correctly'''
    Inner = create_namedtuple('Inner', [('x', int), ('y', int), ('z', int)])

    N1 = create_namedtuple('N1', [('x', Dict[str, int])])
    N2 = create_namedtuple('N2', [('n1s', List[N1])])

    def f1(x: List[Inner]) -> Inner:
        '''Returns the component-wise sum of a sequence of Inner'''
        sums = np.vstack(x).sum(axis=0)
        return Inner(*sums)

    def f2(x: N2) -> N2:
        '''Appends another N1 onto N2'''
        n1 = x.n1s[0]
        n1.x['b'] = 2
        n2 = N2(n1s=[n1, n1])
        return n2

    f1_in = ([Inner(1, 2, 3), ] * 5, )
    f1_out = (5, 10, 15)

    n1 = N1(x={'a': 1})
    n1_out = N1(x={'a': 1, 'b': 2})
    f2_in = N2(n1s=[n1])
    f2_out = N2(n1s=[n1_out, n1_out])

    for func, in_, out in ((f1, f1_in, f1_out), (f2, f2_in, f2_out)):
        _generic_test(func, in_, out)


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


def _build_tf_model(session, data, target):
    '''Builds and iris tensorflow model and returns the prediction tensor'''
    x = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    y = tf.placeholder(shape=[None, 3], dtype=tf.float32)

    layer1 = tf.layers.dense(x, 3, activation=tf.nn.relu)
    logits = tf.layers.dense(layer1, 3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(0.075).minimize(cost)

    init = tf.global_variables_initializer()
    session.run(init)

    for epoch in range(3):
        _, loss = session.run([optimizer, cost], feed_dict={x: data, y: target})

    prediction = tf.argmax(logits, 1)
    return x, y, prediction


def _generic_test(func, in_, out, wrapped_eq=eq, pb_mg_eq=eq, pb_bytes_eq=eq, preload=None, reqs=None):
    '''Reusable wrap test routine with swappable equality functions'''

    model = Model(transform=func)
    model_name = 'my-model'

    with tempfile.TemporaryDirectory() as tdir:
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

        # test all from / as combinations

        assert wrapped_eq(wrapped_model.transform.from_wrapped(trans_in).as_wrapped(), trans_out)
        assert wrapped_eq(wrapped_model.transform.from_pb_msg(trans_in_pb).as_wrapped(), trans_out)
        assert wrapped_eq(wrapped_model.transform.from_pb_bytes(trans_in_pb_bytes).as_wrapped(), trans_out)

        assert pb_mg_eq(wrapped_model.transform.from_wrapped(trans_in).as_pb_msg(), trans_out_pb)
        assert pb_mg_eq(wrapped_model.transform.from_pb_msg(trans_in_pb).as_pb_msg(), trans_out_pb)
        assert pb_mg_eq(wrapped_model.transform.from_pb_bytes(trans_in_pb_bytes).as_pb_msg(), trans_out_pb)

        assert pb_bytes_eq(wrapped_model.transform.from_wrapped(trans_in).as_pb_bytes(), trans_out_pb_bytes)
        assert pb_bytes_eq(wrapped_model.transform.from_pb_msg(trans_in_pb).as_pb_bytes(), trans_out_pb_bytes)
        assert pb_bytes_eq(wrapped_model.transform.from_pb_bytes(trans_in_pb_bytes).as_pb_bytes(), trans_out_pb_bytes)


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

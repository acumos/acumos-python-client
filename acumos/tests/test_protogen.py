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
Provides modeling tests
"""
import tempfile

import pytest
import numpy as np
import pandas as pd

from acumos.modeling import Model, Enum, List, NamedTuple, create_dataframe
from acumos.protogen import _types_equal, _require_unique, _nt2proto, model2proto, compile_protostr
from acumos.exc import AcumosError


def test_type_equality():
    '''Tests that type equality function works as expected'''
    t1 = NamedTuple('T1', [('x', int), ('y', int)])
    t2 = NamedTuple('T1', [('x', int), ('y', float)])
    t3 = NamedTuple('T2', [('x', int), ('y', t1)])
    t4 = NamedTuple('T2', [('x', int), ('y', t2)])
    t5 = NamedTuple('T3', [('x', int), ('y', t1)])
    t6 = NamedTuple('T2', [('x', int), ('y', t1)])
    t7 = NamedTuple('T2', [('x', int), ('z', t1)])

    assert not _types_equal(t1, t2)  # type differs
    assert not _types_equal(t3, t4)  # type differs
    assert not _types_equal(t3, t5)  # name differs
    assert not _types_equal(t3, t7)  # field differs
    assert _types_equal(t3, t6)


def test_require_unique():
    '''Tests that unique types are tested for'''
    t1 = NamedTuple('T1', [('x', int), ('y', int)])
    t2 = NamedTuple('T1', [('x', int), ('y', float)])
    t3 = NamedTuple('T1', [('x', int), ('y', int)])

    with pytest.raises(AcumosError):
        _require_unique((t1, t2, t3))  # t2 is a different definition of T1

    uniq = _require_unique((t1, t3))
    assert len(uniq) == 1
    assert t1 in uniq or t3 in uniq


def test_nt2proto():
    '''Tests the generation of protobuf messages from NamedTuple'''
    Foo = NamedTuple('Foo', [('x', int), ('y', int)])
    Bar = NamedTuple('Bar', [('x', Foo)])

    _nt2proto(Foo, set())

    # dependence on Foo which has not been declared
    with pytest.raises(AcumosError):
        _nt2proto(Bar, set())

    _nt2proto(Bar, {Foo.__name__, })


def test_model2proto():
    '''Tests the generation of protobuf messages from a Model'''
    T1 = NamedTuple('T1', [('x', int), ('y', int)])
    T2 = NamedTuple('T2', [('data', int)])

    Thing = Enum('Thing', 'a b c d e')

    def f1(x: int, y: int) -> int:
        return x + y

    def f2(data: T1) -> T2:
        return T2(data.x + data.y)

    def f3(data: List[Thing]) -> Thing:
        return data[0]

    def f4(data: List[T1]) -> None:
        pass

    def f5(x: List[np.int32]) -> np.int32:
        return np.sum(x)

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    TestDataFrame = create_dataframe('TestDataFrame', df)

    def f6(in_: TestDataFrame) -> None:
        pass

    model = Model(f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6)
    module = 'model'
    package = 'pkg'
    protostr = model2proto(model, package)

    # main test is to make sure that compilation doesn't fail
    with tempfile.TemporaryDirectory() as tdir:
        compile_protostr(protostr, package, module, tdir)


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

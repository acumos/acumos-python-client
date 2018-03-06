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
import pytest

from acumos.modeling import _wrap_function, AcumosError, NamedTuple, Model, NoReturn, Empty, List, Dict
from acumos.protogen import _types_equal


def test_wrap_function():
    '''Tests function wrapper utility'''

    FooIn = NamedTuple('FooIn', [('x', int), ('y', int)])
    FooOut = NamedTuple('FooOut', [('value', int)])

# =============================================================================
#     both args and return need to be wrapped
# =============================================================================
    def foo(x: int, y: int) -> int:
        return x + y

    f, in_, out = _wrap_function(foo)

    assert _types_equal(in_, FooIn)
    assert _types_equal(out, FooOut)
    assert f(FooIn(1, 2)) == FooOut(3)

# =============================================================================
#     function is already considered wrapped
# =============================================================================
    def bar(msg: FooIn) -> FooOut:
        return FooOut(msg.x + msg.y)

    f, in_, out = _wrap_function(bar)

    assert f is bar
    assert _types_equal(in_, FooIn)
    assert _types_equal(out, FooOut)
    assert f(FooIn(1, 2)) == FooOut(3)

# =============================================================================
#     function args need to be wrapped but return is fine
# =============================================================================
    BazIn = NamedTuple('BazIn', [('x', int), ('y', int)])

    def baz(x: int, y: int) -> FooOut:
        return FooOut(x + y)

    f, in_, out = _wrap_function(baz)

    assert _types_equal(in_, BazIn)
    assert _types_equal(out, FooOut)
    assert f(BazIn(1, 2)) == FooOut(3)

# =============================================================================
#     function return needs to be wrapped but args are fine
# =============================================================================
    QuxOut = NamedTuple('QuxOut', [('value', int)])

    def qux(msg: FooIn) -> int:
        return msg.x + msg.y

    f, in_, out = _wrap_function(qux)

    assert _types_equal(in_, FooIn)
    assert _types_equal(out, QuxOut)
    assert f(BazIn(1, 2)) == QuxOut(3)


def test_bad_annotations():
    '''Tests bad annotation scenarios'''

    def f1(x: float):
        pass

    def f2(x):
        pass

    def f3(x) -> float:
        pass

    def f4(x: float, y) -> float:
        pass

    def f5(x: float) -> float:
        pass

    for f in (f1, f2, f3, f4):
        with pytest.raises(AcumosError):
            _wrap_function(f)

    _wrap_function(f5)


def test_model():
    '''Tests Model class'''

    def my_transform(x: int, y: int) -> int:
        '''Docstrings also work'''
        return x + y

    def another_transform(x: int, y: int) -> int:
        return x + y

    model = Model(transform=my_transform, another=another_transform)

    input_type = model.transform.input_type
    output_type = model.transform.output_type

    assert input_type.__name__ == 'TransformIn'
    assert output_type.__name__ == 'TransformOut'

    assert model.transform.inner(1, 1) == 2
    assert model.transform.wrapped(input_type(1, 1)) == output_type(2)

    assert model.transform.description == '''Docstrings also work'''
    assert model.another.description == ''


def test_null_functions():
    '''Tests the wrapping of a function with no arguments and no returns'''
    def f1() -> None:
        pass

    def f2() -> NoReturn:
        pass

    def f3() -> Empty:
        pass

    for f in (f1, f2, f3):
        _, in_, out = _wrap_function(f)
        assert in_ is Empty
        assert out is Empty


def test_reserved_name():
    '''Tests that a reserved NamedTuple name cannot be used'''
    Empty = NamedTuple('Empty', [])

    def foo(x: Empty) -> Empty:
        return Empty()

    with pytest.raises(AcumosError):
        _wrap_function(foo)


def test_nested_defs():
    '''Tests that nested types raise exceptions'''

    def f1(x: List[List[int]]) -> float:
        pass

    def f2(x: List[Dict[str, int]]) -> float:
        pass

    def f3(x: Dict[str, List[int]]) -> float:
        pass

    def f4(x: Dict[str, Dict[str, int]]) -> float:
        pass

    for f in (f1, f2, f3, f4):
        with pytest.raises(AcumosError):
            _wrap_function(f)


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

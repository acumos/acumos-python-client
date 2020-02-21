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
Provides modeling utilities
"""
from inspect import getfullargspec, getdoc, isclass
from typing import NamedTuple, List, Dict, TypeVar, Generic, NewType
from enum import Enum
from collections import namedtuple, OrderedDict

try:
    from typing import NoReturn  # odd import error occurs in Python 3.5 testing
except ImportError:
    NoReturn = None

import numpy as np

from acumos.exc import AcumosError
from acumos.utils import reraise


_NUMPY_PRIMITIVES = {np.int64, np.int32, np.float64, np.float32}
_PYTHON_PRIMITIVES = {int, float, str, bool, bytes}

_VALID_PRIMITIVES = _PYTHON_PRIMITIVES | _NUMPY_PRIMITIVES
_VALID_TYPES = {NamedTuple, List, Dict, Enum} | _VALID_PRIMITIVES

_dtype2prim = {np.dtype(t): t for t in _NUMPY_PRIMITIVES}


class Empty(tuple):
    '''Empty NamedTuple-ish type that consumes any input and returns an empty tuple'''
    __slots__ = ()
    _fields = ()
    _field_types = OrderedDict()

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


_RESERVED_TYPES = {Empty, }
_RESERVED_NAMES = {t.__name__ for t in _RESERVED_TYPES}


RawTypeVar = TypeVar('RawTypeVar', str, bytes, dict)


class Raw(Generic[RawTypeVar]):
    '''Represents raw types that will not result in a generated message'''
    def __init__(self, raw_type: RawTypeVar, metadata: dict, doc: str) -> None:
        self._raw_type = raw_type
        self._metadata = metadata
        self._doc = doc

    @property
    def raw_type(self):
        return self._raw_type

    @property
    def metadata(self):
        return self._metadata

    @property
    def description(self):
        return self._doc


class Model(object):
    '''
    A container of user-provided functions that can be on-boarded as a model in Acumos

    Parameters
    ----------
    kwargs : Arbitrary keyword arguments
        Keys are used as function names and values are user-defined functions
    '''

    def __init__(self, **kwargs):
        if not kwargs:
            raise AcumosError('No functions were provided to Model')

        self._methods = {name: _create_function(func, name) if not isinstance(func, Function) else func
                         for name, func in kwargs.items()}
        for k, v in self._methods.items():
            setattr(self, k, v)

    @property
    def methods(self):
        return self._methods


def _create_function(f, name=None):
    '''Returns an initialized Function object'''
    wrapped, input_type, output_type = _wrap_function(f, name)
    return Function(f, wrapped, input_type, output_type)


class Function(namedtuple('Function', 'inner wrapped input_type output_type')):
    '''Container of original and wrapped functions along with signature metadata'''

    @property
    def description(self):
        doc = getdoc(self.inner)
        return '' if doc is None else doc


def _wrap_function(f, name=None):
    '''Returns a function that has its arguments and return wrapped in NameTuple types'''
    spec = getfullargspec(f)
    anno = spec.annotations

    if 'return' not in anno:
        raise AcumosError("Function {} must have a return annotation".format(f))

    for a in spec.args:
        if a not in anno:
            raise AcumosError("Function argument {} does not have an annotation".format(a))

    if name is None:
        name = f.__name__

    title = ''.join(s for s in name.title().split('_'))

    field_types = [(a, anno[a]) for a in spec.args]
    ret_type = anno['return']

    # check to see if use-defined function consumes and/or produces a raw type, and if so, skip NamedTuple generation
    if _is_raw_type(field_types, ret_type):
        input_type = field_types[0][1]
        output_type = ret_type
        wrapped_f = f
        return wrapped_f, input_type, output_type

    for field_name, field_type in field_types:
        with reraise('Function {} argument {} is invalid', (name, field_name)):
            _assert_valid_type(field_type)

    if ret_type not in (None, NoReturn):
        with reraise('Function {} return type {} is invalid', (name, ret_type)):
            _assert_valid_type(ret_type)

    if _already_wrapped(field_types):
        input_type = field_types[0][1]
        if _is_namedtuple(ret_type):
            output_type = ret_type
            wrapped_f = f
        else:
            output_type = _create_ret_type(title, ret_type)
            wrapped_f = _create_wrapper_ret(f, input_type, output_type)
    else:
        input_type = _create_input_type(title, field_types)
        if _is_namedtuple(ret_type):
            output_type = ret_type
            wrapped_f = _create_wrapper_args(f, input_type, ret_type)
        else:
            output_type = _create_ret_type(title, ret_type)
            wrapped_f = _create_wrapper_both(f, input_type, output_type)

    with reraise('Function {} wrapped input type is invalid', (name, )):
        _assert_valid_type(input_type)

    with reraise('Function {} wrapped output type is invalid', (name, )):
        _assert_valid_type(output_type)

    return wrapped_f, input_type, output_type


def _is_raw_type(field_types, ret_type):
    '''Returns True if field type and ret_type are one of the supported raw types and the user-defined function only consumes and/or produces one raw type'''
    if not field_types or not ret_type:
        return False

    if not hasattr(field_types[0][1], '__supertype__') and not hasattr(ret_type, '__supertype__'):
        return False

    try:
        '''check if output and input are both unstructured'''
        assert hasattr(field_types[0][1], '__supertype__')
        assert hasattr(ret_type, '__supertype__')
    except AssertionError:
        raise AcumosError("Usage of unstructured type is only possible when the user-defined function consumes or produces unstructured type for both input and output")

    return len(field_types) == 1 and type(field_types[0][1].__supertype__) == Raw and type(ret_type.__supertype__) == Raw


def _already_wrapped(field_types):
    '''Returns True if the field types are already considered wrapped'''
    return len(field_types) == 1 and _is_namedtuple(field_types[0][1])


def _is_namedtuple(t):
    '''Returns True if type `t` is a NamedTuple type'''
    return isclass(t) and issubclass(t, tuple) and hasattr(t, '_field_types')


def _is_subclass(c, t):
    '''Returns True if c is a subclass of t'''
    return isclass(c) and issubclass(c, t)


def _create_input_type(name, field_types):
    '''Generates a NamedTuple for input arguments'''
    return NamedTuple("{}In".format(name), field_types) if field_types else Empty


def _create_ret_type(name, ret_type):
    '''Generates a NamedTuple for a function return'''
    return NamedTuple("{}Out".format(name), [('value', ret_type)]) if ret_type not in (None, NoReturn) else Empty


def _create_wrapper_both(f, input_type, output_type):
    '''Returns a wrapped function that accepts and returns NamedTuple types'''
    def wrapped(in_: input_type) -> output_type:
        ret = f(*in_)
        return output_type(ret)
    return wrapped


def _create_wrapper_args(f, input_type, output_type):
    '''Returns a wrapped function that accepts and returns NamedTuple types'''
    def wrapped(in_: input_type) -> output_type:
        return f(*in_)
    return wrapped


def _create_wrapper_ret(f, input_type, output_type):
    '''Returns a wrapped function that accepts and returns NamedTuple types'''
    def wrapped(in_: input_type) -> output_type:
        ret = f(in_)
        return output_type(ret)
    return wrapped


def _assert_valid_type(t, container=None):
    '''Raises AcumosError if the input type contains an invalid type'''
    if t in _VALID_PRIMITIVES:
        pass

    elif _is_namedtuple(t):
        if t.__name__ in _RESERVED_NAMES and t not in _RESERVED_TYPES:
            raise AcumosError("NamedTuple {} cannot use a reserved name: {}".format(t, _RESERVED_NAMES))

        for tt in t._field_types.values():
            _assert_valid_type(tt)

    elif _is_subclass(t, List):
        if container is not None:
            raise AcumosError("List types cannot be nested within {} types. Use NamedTuple instead".format(container.__name__))

        _assert_valid_type(t.__args__[0], container=List)

    elif _is_subclass(t, Dict):
        if container is not None:
            raise AcumosError("Dict types cannot be nested within {} types. Use NamedTuple instead".format(container.__name__))

        key_type, value_type = t.__args__

        if key_type is not str:
            raise AcumosError('Dict keys must be str type')

        _assert_valid_type(value_type, container=Dict)

    elif _is_subclass(t, Enum):
        pass

    else:
        raise AcumosError("Type {} is not one of the supported types: {}".format(t, _VALID_TYPES))


def create_dataframe(name, df):
    '''Returns a NamedTuple type corresponding to a pandas DataFrame instance'''
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise AcumosError('Input `df` must be a pandas.DataFrame')

    dtypes = list(df.dtypes.iteritems())
    for field_name, dtype in dtypes:
        if dtype not in _dtype2prim:
            raise AcumosError("DataFrame column '{}' has an unsupported type '{}'. Supported types are: {}".format(field_name, dtype, _NUMPY_PRIMITIVES))

    field_types = [(n, List[_dtype2prim[dt]]) for n, dt in dtypes]
    df_type = NamedTuple(name, field_types)
    return df_type


def create_namedtuple(name, field_types):
    '''Returns a NamedTuple type'''
    return NamedTuple(name, field_types)


def new_type(raw_type, name, metadata=None, doc=None):
    '''Returns a user specified raw type'''
    return NewType(name, Raw(raw_type, metadata, doc))

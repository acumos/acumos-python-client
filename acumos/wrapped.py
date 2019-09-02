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
Provides wrapped model utilities
"""
import sys
from zipfile import ZipFile
from os import listdir
from os.path import isfile, isdir, join as path_join

from google.protobuf.json_format import Parse as ParseJson, ParseDict, MessageToJson, MessageToDict

from acumos.modeling import _is_namedtuple, List, Dict
from acumos.pickler import AcumosContextManager, load_model as _load_model
from acumos.utils import load_module
from acumos.exc import AcumosError


def load_model(path):
    '''Returns a WrappedModel previously dumped to `path`'''
    model_dir = _infer_model_dir(path)
    with AcumosContextManager(model_dir) as c:
        _extend_syspath(c)  # extend path before we unpickle user model

        with open(c.build_path('model.pkl'), 'rb') as f:
            model = _load_model(f)

        pkg = c.parameters['protobuf_package']
        module_path = c.build_path(path_join('scripts', 'acumos_gen', pkg, 'model_pb2.py'))
        module = load_module('model_pb2', module_path)
        return WrappedModel(model, module)


def _infer_model_dir(path):
    '''Returns an absolute path to the model dir. Unzips the model archive if `path` contains it'''
    model_zip_path = path_join(path, 'model.zip')
    if isfile(model_zip_path):
        model_dir = path_join(path, 'model')
        zip_file = ZipFile(model_zip_path)
        zip_file.extractall(model_dir)
    else:
        model_dir = path

    pkl_path = path_join(model_dir, 'model.pkl')
    if not isfile(pkl_path):
        raise AcumosError("Provided path {} does not contain an Acumos model".format(path))

    return model_dir


def _extend_syspath(context):
    '''Adds user-provided packages to the system path'''
    provided_abspath = context.build_path(path_join('scripts', 'user_provided'))
    if provided_abspath not in sys.path:
        sys.path.append(provided_abspath)

    for pkg_name in listdir(provided_abspath):
        pkg_abspath = path_join(provided_abspath, pkg_name)
        if not isdir(pkg_abspath):
            continue

        if pkg_abspath not in sys.path:
            sys.path.append(pkg_abspath)


class WrappedModel(object):
    '''Container of WrappedFunction objects'''

    def __init__(self, model, module):
        self._methods = {name: WrappedFunction(func, module) for name, func in model.methods.items()}
        for k, v in self._methods.items():
            setattr(self, k, v)

    @property
    def methods(self):
        return self._methods


class WrappedFunction(object):
    '''A function wrapper with various consumption options'''

    def __init__(self, func, module):
        self._func = func
        self._module = module
        self._input_type = func.input_type
        self._output_type = func.output_type
        self._pb_input_type = getattr(module, func.input_type.__name__) if _is_namedtuple(self._input_type) else None
        self._pb_output_type = getattr(module, func.output_type.__name__) if _is_namedtuple(self._output_type) else None

    def from_pb_bytes(self, pb_bytes_in):
        '''Consumes a binary Protobuf message and returns a WrappedResponse object'''
        pb_msg_in = self._pb_input_type.FromString(pb_bytes_in)
        return self.from_pb_msg(pb_msg_in)

    def from_pb_msg(self, pb_msg_in):
        '''Consumes a Protobuf message object and returns a WrappedResponse object'''
        wrapped_in = _unpack_pb_msg(self._input_type, pb_msg_in)
        return self.from_wrapped(wrapped_in)

    def from_native(self, *args, **kwargs):
        '''Consumes the original inner function arguments and returns a WrappedResponse object'''
        wrapped_in = self._input_type(*args, **kwargs)
        return self.from_wrapped(wrapped_in)

    def from_wrapped(self, wrapped_in):
        '''Consumes a NamedTuple wrapped type and returns a WrappedResponse object'''
        wrapped_out = self._func.wrapped(wrapped_in)
        return WrappedResponse(wrapped_out, self._module, self._pb_output_type)

    def from_dict(self, dict_in):
        '''Consumes a dict and returns a WrappedResponse object'''
        pb_msg_in = ParseDict(dict_in, self._pb_input_type())
        return self.from_pb_msg(pb_msg_in)

    def from_json(self, json_in):
        '''Consumes a json str and returns a WrappedResponse object'''
        pb_msg_in = ParseJson(json_in, self._pb_input_type())
        return self.from_pb_msg(pb_msg_in)

    def from_raw(self, raw_in):
        '''Consumes a raw type data and returns a WrappedResponse object'''
        raw_out = self._func.inner(raw_in)
        return WrappedResponse(raw_out, None, None)

    @property
    def pb_input_type(self):
        return self._pb_input_type

    @property
    def pb_output_type(self):
        return self._pb_output_type


class WrappedResponse(object):
    '''A WrappedFunction response with various return options'''

    def __init__(self, resp, module, pb_output_type):
        self._resp = resp
        self._module = module
        self._pb_output_type = pb_output_type

    def as_pb_bytes(self):
        '''Returns a Protobuf binary string representation of the model response'''
        return self.as_pb_msg().SerializeToString()

    def as_pb_msg(self):
        '''Returns a Protobuf message representation of the model response'''
        return _pack_pb_msg(self._resp, self._module)

    def as_wrapped(self):
        '''Returns a Python NamedTuple representation of the model response'''
        return self._resp

    def as_dict(self):
        '''Returns a dict representation of the model response'''
        pb_msg_out = self.as_pb_msg()
        return MessageToDict(pb_msg_out, self._pb_output_type())

    def as_json(self):
        '''Returns a json str representation of the model response'''
        pb_msg_out = self.as_pb_msg()
        return MessageToJson(pb_msg_out, self._pb_output_type(), indent=0)

    def as_raw(self):
        '''Returns a raw data type representation of the model response'''
        return self._resp


def _pack_pb_msg(wrapped_in, module):
    '''Returns a protobuf message object from a NamedTuple instance'''
    wrapped_type = type(wrapped_in)
    field_types = wrapped_type._field_types
    pb_type = getattr(module, wrapped_type.__name__)
    return pb_type(**{f: _set_pb_value(field_types[f], v, module) for f, v in zip(wrapped_in._fields, wrapped_in)})


def _set_pb_value(wrapped_type, value, module):
    '''Recursively traverses the NamedTuple instance to ensure nested NamedTuples become protobuf messages'''
    if _is_namedtuple(wrapped_type):
        return _pack_pb_msg(value, module)

    elif issubclass(wrapped_type, Dict):
        _, val_type = wrapped_type.__args__
        if _is_namedtuple(val_type):
            return {k: _pack_pb_msg(v, module) for k, v in value.items()}

    elif issubclass(wrapped_type, List):
        list_type = wrapped_type.__args__[0]
        if _is_namedtuple(list_type):
            return [_pack_pb_msg(v, module) for v in value]

    return value


def _unpack_pb_msg(input_type, pb_msg):
    '''Returns a NamedTuple from protobuf message'''
    values = {f: _get_pb_value(t, getattr(pb_msg, f)) for f, t in input_type._field_types.items()}
    return input_type(**values)


def _get_pb_value(wrapped_type, pb_value):
    '''Recursively traverses the protobuf message to ensure nested messages become NamedTuples'''
    if _is_namedtuple(wrapped_type):
        return _unpack_pb_msg(wrapped_type, pb_value)

    elif issubclass(wrapped_type, Dict):
        _, val_type = wrapped_type.__args__
        if _is_namedtuple(val_type):
            return {k: _unpack_pb_msg(val_type, v) for k, v in pb_value.items()}

    elif issubclass(wrapped_type, List):
        list_type = wrapped_type.__args__[0]
        if _is_namedtuple(list_type):
            return [_unpack_pb_msg(list_type, v) for v in pb_value]

    return pb_value

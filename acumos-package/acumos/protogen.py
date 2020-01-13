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
Provides protobuf generation utilities
"""
import shutil
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory
from collections import defaultdict
from itertools import chain as iterchain
from os.path import dirname, isfile, join as path_join
from os import makedirs

import numpy as np

from acumos.exc import AcumosError
from acumos.modeling import List, Dict, Enum, _is_namedtuple
from acumos.utils import namedtuple_field_types


_PROTO_SYNTAX = 'syntax = "proto3";'

_package_template = 'package {name};'

_message_template = '''
message {name} {{
{msg_def}
}}'''

_enum_template = '''
enum {name} {{
{enum_def}
}}'''

_service_template = '''
service {name} {{
{service_def}
}}'''

_rpc_template = 'rpc {name} ({msg_in}) returns ({msg_out});'


class NestedTypeError(AcumosError):
    pass


_type_lookup = {
    int: 'int64',
    str: 'string',
    float: 'double',
    bool: 'bool',
    bytes: 'bytes',
    np.int64: 'int64',
    np.int32: 'int32',
    np.float32: 'float',
    np.float64: 'double',
    np.bool: 'bool'}


def compile_protostr(proto_str, package_name, module_name, out_dir):
    '''Compiles a Python module from a protobuf definition str and returns the module abspath'''
    _assert_protoc()

    with TemporaryDirectory() as tdir:
        protopath = path_join(tdir, package_name, "{}.proto".format(module_name))

        makedirs(dirname(protopath))
        with open(protopath, 'w') as f:
            f.write(proto_str)

        cmd = "protoc --python_out {tdir} --proto_path {tdir} {protopath}".format(tdir=tdir, protopath=protopath).split()
        p = Popen(cmd, stderr=PIPE)
        _, err = p.communicate()
        if p.returncode != 0:
            raise AcumosError("A failure occurred while generating source code from protobuf: {}".format(err))

        gen_module_name = "{}_pb2.py".format(module_name)
        gen_module_path = path_join(tdir, package_name, gen_module_name)
        if not isfile(gen_module_path):
            raise AcumosError("An unknown failure occurred while generating Python module {}".format(gen_module_path))

        out_module_path = path_join(out_dir, gen_module_name)
        shutil.copy(gen_module_path, out_module_path)
        return out_module_path


def _assert_protoc():
    '''Raises an AcumosError if protoc is not found'''
    if shutil.which('protoc') is None:
        raise AcumosError('The protocol buffers compiler `protoc` was not found. Verify that it is installed and visible in $PATH')


def model2proto(model, package_name):
    '''Converts a Model object to a protobuf schema string'''
    all_types = (iterchain(*(iterchain(_proto_iter(f.input_type),
                                       _proto_iter(f.output_type)) for f in model.methods.values())))

    unique_types = _require_unique(all_types)
    type_names = set(t.__name__ for t in unique_types)

    msg_defs = tuple(_nt2proto(t, type_names) if _is_namedtuple(t) else _enum2proto(t) for t in unique_types)
    service_def = _gen_service(model)
    package_def = _package_template.format(name=package_name)

    defs = (_PROTO_SYNTAX, package_def, service_def) + msg_defs
    return '\n'.join(defs)


def _proto_iter(nt):
    '''Recursively yields all types contained within the NamedTuple relevant to protobuf gen'''
    if _is_namedtuple(nt):
        yield nt

        for t in nt._field_types.values():
            if _is_namedtuple(t):
                yield from _proto_iter(t)
            elif issubclass(t, Enum):
                yield t
            elif issubclass(t, List) or issubclass(t, Dict):
                for tt in t.__args__:
                    yield from _proto_iter(tt)


def _require_unique(types):
    '''Returns a list of unique types. Raises AcumosError if named types are not uniquely defined'''
    dd = defaultdict(list)
    for t in types:
        dd[t.__name__].append(t)

    for n, l in dd.items():
        if len(l) > 1 and not all(_types_equal(l[0], t) for t in l[1:]):
            raise AcumosError("Multiple definitions found for type {}: {}".format(n, l))

    return [l[0] for l in dd.values()]


def _types_equal(t1, t2):
    '''Returns True if t1 and t2 types are equal. Can't override __eq__ on NamedTuple unfortunately.'''
    if _is_namedtuple(t1) and _is_namedtuple(t2):
        names_match = t1.__name__ == t2.__name__
        ft1, ft2 = namedtuple_field_types(t1), namedtuple_field_types(t2)
        keys_match = ft1.keys() == ft2.keys()
        values_match = all(_types_equal(v1, v2) for v1, v2 in zip(ft1.values(), ft2.values()))
        return names_match and keys_match and values_match

    if issubclass(t1, Enum) and issubclass(t2, Enum):
        names_match = t1.__name__ == t2.__name__
        enums_match = [(e.name, e.value) for e in t1] == [(e.name, e.value) for e in t2]
        return names_match and enums_match

    else:
        return t1 == t2


def _nt2proto(nt, type_names):
    '''Converts a NamedTuple definition to a protobuf schema str'''
    msg_def = '\n'.join(_field2proto(field, type_, index, type_names)
                        for index, (field, type_) in enumerate(namedtuple_field_types(nt).items(), 1))
    return _message_template.format(name=nt.__name__, msg_def=msg_def)


def _enum2proto(enum):
    '''Converts an Enum definition to a protobuf schema str'''
    names, values = zip(*((e.name, e.value) for e in enum))
    return _gen_enum(names, enum.__name__, values)


def _gen_enum(enums, name, values=None):
    '''Produces an enum protobuf definition. An UNKNOWN enum is added if values are provided without a 0'''
    if values is not None:
        if 0 not in values:
            enums = ('UNKNOWN', ) + tuple(enums)
            values = (0, ) + tuple(values)

        enums, values = zip(*sorted(zip(enums, values), key=lambda x: x[1]))

        # the first enum value must be zero in proto3
        zidx = values.index(0)
        enums = enums[zidx:] + enums[:zidx]
        values = values[zidx:] + values[:zidx]
    else:
        values = range(len(enums))

    enum_def = '\n'.join(("  {} = {};".format(name, idx) for name, idx in zip(enums, values)))
    return _enum_template.format(name=name, enum_def=enum_def)


def _field2proto(name, type_, index, type_names, rjust=None):
    '''Returns a protobuf schema field str from a NamedTuple field'''
    string = None

    if type_ in _type_lookup:
        string = "{} {} = {};".format(_type2proto(type_), name, index)

    elif _is_namedtuple(type_) or issubclass(type_, Enum):
        tn = type_.__name__
        if tn not in type_names:
            raise AcumosError("Could not build protobuf field using unknown custom type {}".format(tn))
        string = "{} {} = {};".format(tn, name, index)

    elif issubclass(type_, List):
        inner = type_.__args__[0]
        if _is_container(inner):
            raise NestedTypeError("Nested container {} is not yet supported; try using NamedTuple instead".format(type_))
        string = "repeated {}".format(_field2proto(name, inner, index, type_names, 0))

    elif issubclass(type_, Dict):
        k, v = type_.__args__
        if any(map(_is_container, (k, v))):
            raise NestedTypeError("Nested container {} is not yet supported; try using NamedTuple instead".format(type_))
        string = "map<{}, {}> {} = {};".format(_type2proto(k), _type2proto(v), name, index)

    if string is None:
        raise AcumosError("Could not build protobuf field due to unsupported type {}".format(type_))

    if rjust is None:
        rjust = len(string) + 2

    return string.rjust(rjust, ' ')


def _is_container(t):
    return issubclass(t, Dict) or issubclass(t, List)


def _type2proto(t):
    '''Returns a string corresponding to the protobuf type'''
    if t in _type_lookup:
        return _type_lookup[t]
    elif _is_namedtuple(t) or issubclass(t, Enum):
        return t.__name__
    else:
        raise AcumosError("Unknown protobuf mapping for type {}".format(t))


def _gen_service(model, name='Model'):
    '''Returns a protobuf service definition string'''
    rpc_comps = ((n, f.input_type.__name__, f.output_type.__name__) for n, f in model.methods.items() if _is_namedtuple(f.input_type))
    rpc_defs = '\n'.join(_gen_rpc(*comps) for comps in rpc_comps)
    return _service_template.format(name=name, service_def=rpc_defs)


def _gen_rpc(name, in_, out):
    '''Returns a protobuf rpc definition string'''
    rpc = _rpc_template.format(name=name, msg_in=in_, msg_out=out)
    return rpc.rjust(len(rpc) + 2)

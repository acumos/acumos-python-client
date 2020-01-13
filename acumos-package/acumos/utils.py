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
Provides model wrapping utilities
"""
from importlib.util import spec_from_file_location, module_from_spec
import os
import inspect
import contextlib
from collections import OrderedDict

from acumos.exc import AcumosError


def namedtuple_field_types(nt):
    '''Returns an OrderedDict corresponding to NamedTuple field types'''
    field_types = nt._field_types
    return OrderedDict((field, field_types[field]) for field in nt._fields)


def load_module(fullname, path):
    '''Imports and returns a module from path for Python 3.5+'''
    spec = spec_from_file_location(fullname, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_artifact(*path, module, mode):
    '''Artifact loader helper'''
    with open(os.path.join(*path), mode) as f:
        return module.load(f)


def dump_artifact(*path, data, module, mode):
    '''Artifact saver helper'''
    with open(os.path.join(*path), mode) as f:
        if module is None:
            f.write(data)
        else:
            module.dump(data, f)


def get_qualname(o):
    if inspect.isclass(o):
        return "{}.{}".format(o.__module__, o.__name__)
    else:
        return get_qualname(o.__class__)


@contextlib.contextmanager
def reraise(prefix, prefix_args):
    '''Reraises an exception with a more informative prefix'''
    try:
        yield
    except AcumosError as e:
        raise AcumosError("{}: {}".format(prefix.format(*prefix_args), e)).with_traceback(e.__traceback__)

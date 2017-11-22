# -*- coding: utf-8 -*-
"""
Provides model wrapping utilities
"""
import sys
import os
import re
import inspect
import contextlib

from acumos.exc import AcumosError


def _load_module_py33(fullname, path):
    '''Imports and returns a module from path for Python 3.3-3.4'''
    from importlib.machinery import SourceFileLoader
    module = SourceFileLoader(fullname, path).load_module()
    return module


def _load_module_py35(fullname, path):
    '''Imports and returns a module from path for Python 3.5+'''
    import importlib.util
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(fullname, path):
    '''Imports and returns a module from path'''
    ver_info = sys.version_info
    if (3, 3) <= ver_info < (3, 5):
        return _load_module_py33(fullname, path)
    elif (3, 5) <= ver_info:
        return _load_module_py35(fullname, path)
    else:
        raise AcumosError("Attempted to import Python module from path, but Python {} is not supported".format(ver_info))


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


def sanitize_field(name):
    return re.sub(r"\W+", '_', name)


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

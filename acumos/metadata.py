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
Provides metadata generation utilities
"""
import importlib
import sys
from types import ModuleType
from pkg_resources import get_distribution, DistributionNotFound
from os.path import basename, normpath, sep as pathsep

from acumos.exc import AcumosError


BUILTIN_MODULE_NAMES = set(sys.builtin_module_names)
PACKAGE_DIRS = {'site-packages', 'dist-packages'}
SCHEMA_VERSION = '0.4.0'
_SCHEMA = "acumos.schema.model:{}".format(SCHEMA_VERSION)

# known package mappings because Python packaging is madness
_REQ_MAP = {
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow'
}


class Requirements(object):
    '''
    A collection of optional user-provided Python requirement metadata

    Parameters
    ----------
    reqs : Sequence[str]
        A sequence of pip-installable Python package names
    req_map : Dict[str, str] or Dict[module, str]
        A corrective mapping of Python modules to pip-installable package names. For example
        `req_map={'sklearn': 'scikit-learn'}` or `import sklearn; req_map={sklearn: 'scikit-learn'}`.
    packages : Sequence[str]
        A sequence of paths to directories containing Python packages (i.e. Python packages). The
        directories will be copied along with your model and added to the PYTHONPATH at runtime.
    '''

    __slots__ = ('reqs', 'req_map', 'packages')

    def __init__(self, reqs=None, req_map=None, packages=None):
        self.reqs = set() if reqs is None else _safe_set(reqs)
        self.req_map = dict() if req_map is None else req_map.copy()
        self.req_map.update(_REQ_MAP)
        self.packages = set() if packages is None else _safe_set(packages)

    @property
    def package_names(self):
        return set(map(basename, self.packages))


def _safe_set(input_):
    '''Safely returns a set from input'''
    return {input_, } if isinstance(input_, str) else set(input_)


def create_model_meta(model, name, requirements, encoding='protobuf'):
    '''Returns a model metadata dictionary'''
    return {'schema': _SCHEMA,
            'runtime': _create_runtime(requirements, encoding),
            'name': name,
            'methods': {name: {'input': f.input_type.__name__,
                               'output': f.output_type.__name__,
                               'description': f.description} for name, f in model.methods.items()}}


def _create_runtime(requirements, encoding='protobuf'):
    '''Returns a runtime dict'''
    req_set = _create_requirement_set(requirements)
    req_tuples = _req_iter(req_set, requirements.package_names)
    return {'name': 'python',
            'encoding': encoding,
            'version': '.'.join(map(str, sys.version_info[:3])),
            'dependencies': _create_requirements(req_tuples)}


def _create_requirement_set(requirements):
    '''Returns a set of requirement names'''
    req_names = (n for n in map(_get_requirement_name, requirements.reqs) if not _in_stdlib(n))
    req_map = {_get_requirement_name(k): v for k, v in requirements.req_map.items()}
    mapped_reqs = {req_map.get(r, r) for r in req_names}
    return mapped_reqs


def _get_requirement_name(req):
    '''Returns the str name of the requirement'''
    if isinstance(req, ModuleType):
        name = req.__name__
    elif isinstance(req, str):
        name = req
    else:
        raise AcumosError("Requirement {} is invalid; must be ModuleType or string".format(req))
    return name


def _req_iter(req_names, package_names):
    '''Yields (package, version) tuples'''
    for req_name in req_names:
        try:
            yield str(get_distribution(req_name).as_requirement()).split('==')
        except DistributionNotFound:
            if req_name not in package_names:
                raise AcumosError("Module {} was detected as a dependency, but not found as a pip installed package. Use acumos.session.Requirements to declare custom packages or map module names to pip-installable names (e.g. Requirements(req_map=dict(cv2='opencv-python')) )".format(req_name))


def _in_stdlib(module_name):
    '''Returns True if the package name is part of the standard library (not airtight)'''
    if module_name in BUILTIN_MODULE_NAMES:
        return True

    try:
        install_path = importlib.import_module(module_name).__file__
    except ImportError:
        return False

    if not install_path.startswith(sys.prefix):
        return False

    dirs = set(normpath(install_path).split(pathsep))
    return not PACKAGE_DIRS & dirs


def _create_requirements(req_tuples):
    '''Returns a dict containing model implementation metadata'''
    return {
        'pip': {
            'indexes': [],
            'requirements': [{'name': n, 'version': v} for n, v in req_tuples]
        },
        'conda': {
            'channels': [],
            'requirements': []
        }
    }

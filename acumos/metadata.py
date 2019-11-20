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
from acumos.modeling import _is_namedtuple


BUILTIN_MODULE_NAMES = set(sys.builtin_module_names)
PACKAGE_DIRS = {'site-packages', 'dist-packages'}
SCHEMA_VERSION = '0.6.0'
_SCHEMA = "acumos.schema.model:{}".format(SCHEMA_VERSION)
_MEDIA_TYPES = {str: 'text/plain', bytes: 'application/octet-stream', dict: 'application/json', 'protobuf': 'application/vnd.google.protobuf'}

# known package mappings because Python packaging is madness
_REQ_MAP = {
    'sklearn': 'scikit-learn',
}


class Options(object):
    '''
    A collection of options that users may wish to specify along with their Acumos model

    Parameters
    ----------
    create_microservice : bool, optional
        If True, instructs the Acumos platform to eagerly build the model microservice
    license : str, optional
        A license to include with the Acumos model. This parameter may either be a path to a license
        file, or a string containing the license content.
    '''
    __slots__ = ('create_microservice', 'license')

    def __init__(self, create_microservice=True, license=None):
        self.create_microservice = create_microservice
        self.license = license


class Requirements(object):
    '''
    A collection of optional user-provided Python requirement metadata

    Parameters
    ----------
    reqs : Sequence[str], optional
        A sequence of pip-installable Python package names
    req_map : Dict[str, str] or Dict[module, str], optional
        A corrective mapping of Python modules to pip-installable package names. For example
        `req_map={'sklearn': 'scikit-learn'}` or `import sklearn; req_map={sklearn: 'scikit-learn'}`.
    packages : Sequence[str], optional
        A sequence of paths to Python packages (i.e. directories with an __init__.py). Provided Python
        packages will be copied along with your model and added to the PYTHONPATH at runtime.
    scripts : Sequence[str], optional
        A sequence of paths to Python scripts. A path can point to a Python script, or directory
        containing Python scripts. If a directory is provided, all Python scripts within the directory
        will be included. Provided Python scripts will be copied along with your model and added to the
        PYTHONPATH at runtime.
    '''

    __slots__ = ('reqs', 'req_map', 'packages', 'scripts')

    def __init__(self, reqs=None, req_map=None, packages=None, scripts=None):
        self.reqs = set() if reqs is None else _safe_set(reqs)
        self.req_map = dict() if req_map is None else req_map.copy()
        self.req_map.update(_REQ_MAP)
        self.packages = set() if packages is None else _safe_set(packages)
        self.scripts = set() if scripts is None else _safe_set(scripts)

    @property
    def package_names(self):
        return frozenset(basename(p.rstrip(pathsep)) for p in self.packages)


def _safe_set(input_):
    '''Safely returns a set from input'''
    return {input_, } if isinstance(input_, str) else set(input_)


def create_model_meta(model, name, requirements, encoding='protobuf'):
    '''Returns a model metadata dictionary'''
    return {'schema': _SCHEMA,
            'runtime': _create_runtime(requirements, encoding),
            'name': name,
            'methods': {name: {'input': {'name': f.input_type.__name__,
                                         'media_type': [_MEDIA_TYPES[encoding if _is_namedtuple(f.input_type) else f.input_type.__supertype__._raw_type]],
                                         'metadata': {} if _is_namedtuple(f.input_type) else f.input_type.__supertype__._metadata,
                                         'description': '' if _is_namedtuple(f.input_type) else f.input_type.__supertype__._doc},
                               'output': {'name': f.output_type.__name__,
                                          'media_type': [_MEDIA_TYPES[encoding if _is_namedtuple(f.output_type) else f.output_type.__supertype__._raw_type]],
                                          'metadata': {} if _is_namedtuple(f.output_type) else f.output_type.__supertype__._metadata,
                                          'description': '' if _is_namedtuple(f.output_type) else f.output_type.__supertype__._doc},
                               'description': f.description} for name, f in model.methods.items()}}


def _create_runtime(requirements, encoding='protobuf'):
    '''Returns a runtime dict'''
    reqs = _gather_requirements(requirements)
    return {'name': 'python',
            'version': '.'.join(map(str, sys.version_info[:3])),
            'dependencies': _create_dependencies(reqs)}


def _gather_requirements(requirements):
    '''Yields (name, version) tuples of required 3rd party Python packages'''
    for req_name in _filter_requirements(requirements):
        yield _get_distribution(req_name)


def _filter_requirements(requirements):
    '''Returns a set required 3rd party Python package names'''
    # first get all non-stdlib requirement names
    req_names = (n for n in map(_get_requirement_name, requirements.reqs) if not _in_stdlib(n))

    # then apply any user-provided requirement mappings
    req_map = {_get_requirement_name(k): v for k, v in requirements.req_map.items()}
    mapped_reqs = {req_map.get(r, r) for r in req_names}

    # finally remove user-provided custom package names, as they won't exist in pip
    filtered_reqs = mapped_reqs - requirements.package_names
    return filtered_reqs


def _get_requirement_name(req):
    '''Returns the str name of a requirement'''
    if isinstance(req, ModuleType):
        name = req.__name__
    elif isinstance(req, str):
        name = req
    else:
        raise AcumosError("Requirement {} is invalid; must be ModuleType or string".format(req))
    return name


def _get_distribution(req_name):
    '''Returns (name, version) tuple given a requirement'''
    try:
        return str(get_distribution(req_name).as_requirement()).split('==')
    except DistributionNotFound:
        raise AcumosError("Module {} was detected as a dependency, but not found as a pip installed package. Use acumos.session.Requirements to declare custom packages or map module names to pip-installable names (e.g. Requirements(req_map=dict(PIL='pillow')) )".format(req_name))


def _in_stdlib(module_name):
    '''Returns True if the package name is part of the standard library (not airtight)'''
    if module_name in BUILTIN_MODULE_NAMES:
        return True

    try:
        install_path = importlib.import_module(module_name).__file__
    except ImportError:
        return False

    if not install_path.startswith(normpath(sys.base_prefix)):
        return False

    dirs = set(normpath(install_path).split(pathsep))
    return not PACKAGE_DIRS & dirs


def _create_dependencies(req_tuples):
    '''Returns a dict containing model dependency metadata'''
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

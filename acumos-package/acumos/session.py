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
Provides a Acumos session for pushing and dumping models
"""
import random
import string
import shutil
import json
import requests
import fnmatch
import warnings
from contextlib import contextmanager, ExitStack
from tempfile import TemporaryDirectory
from os import walk, mkdir
from os.path import extsep, exists, abspath, dirname, isdir, isfile, expanduser, relpath, basename, join as path_join
from pathlib import Path
from collections import namedtuple
from glob import glob

from acumos.pickler import AcumosContextManager, dump_model
from acumos.metadata import create_model_meta, Requirements, Options
from acumos.utils import dump_artifact, get_qualname
from acumos.exc import AcumosError
from acumos.protogen import model2proto, compile_protostr
from acumos.logging import get_logger
from acumos.modeling import Model
from acumos.auth import get_jwt, clear_jwt


logger = get_logger(__name__)

_LICENSE_NAME = 'license.json'
_PYEXT = "{}py".format(extsep)
_PYGLOB = "*{}".format(_PYEXT)

_ServerResponse = namedtuple('ServerResponse', 'status_code reason text')
_DEPR_MSG = ('Usage of `auth_api` is deprecated; provide an onboarding token instead. '
             'See https://pypi.org/project/acumos/ for more information.')


class AcumosSession(object):
    '''
    A session that enables onboarding models to Acumos

    Parameters
    ----------
    push_api : str
        The full URL to the Acumos onboarding server upload API
    auth_api : str
        The full URL to the Acumos onboarding server authentication API.

        .. deprecated:: 0.7.1
            Users should provide an onboarding token instead of username and password credentials.
    '''

    def __init__(self, push_api=None, auth_api=None):
        self.push_api = push_api
        self.auth_api = auth_api

        if auth_api is not None:
            warnings.warn(_DEPR_MSG, DeprecationWarning, stacklevel=2)

    def push(self, model, name, requirements=None, extra_headers=None, options=None):
        '''
        Pushes a model to Acumos

        Parameters
        ----------
        model : ``acumos.modeling.Model``
            An Acumos model instance
        name : str
            The name of your model
        requirements : ``acumos.metadata.Requirements``, optional
            Additional Python dependencies that you can optionally specify
        extra_headers : dict, optional
            Additonal HTTP headers included in the POST to the Acumos onboarding server
        options : ``acumos.metadata.Options``, optional
            Additional Acumos options that you can optionally specify
        '''
        _assert_valid_input(model, requirements)
        _assert_valid_api('push_api', self.push_api, True)
        _assert_valid_api('auth_api', self.auth_api, False)
        options = _validate_options(options)

        with _dump_model(model, name, requirements) as dump_dir:
            _push_model(dump_dir, self.push_api, self.auth_api, options, extra_headers=extra_headers)

    def dump(self, model, name, outdir, requirements=None):
        '''
        Creates a directory located at ``outdir/name`` containing Acumos model artifacts

        Parameters
        ----------
        model : ``acumos.modeling.Model``
            An Acumos model instance
        name : str
            The name of your model
        outdir : str
            The directory or folder to save your model .zip to
        requirements : ``acumos.metadata.Requirements``, optional
            Additional Python dependencies that you can optionally specify
        '''
        _assert_valid_input(model, requirements)

        with _dump_model(model, name, requirements) as dump_dir:
            _copy_dir(dump_dir, expanduser(outdir), name)


def _validate_options(options):
    '''Validates and returns an `Options` object'''
    if options is None:
        options = Options()
    elif not isinstance(options, Options):
        raise AcumosError('The `options` parameter must be of type `acumos.metadata.Options`')
    return options


def _assert_valid_input(model, requirements):
    '''Raises AcumosError if inputs are invalid'''
    if not isinstance(model, Model):
        raise AcumosError("Input `model` must be of type {}".format(get_qualname(Model)))

    if requirements is not None and not isinstance(requirements, Requirements):
        raise AcumosError("Input `requirements` must be of type {}".format(get_qualname(Requirements)))


def _assert_valid_api(param, api, required):
    '''Raises AcumosError if an api is invalid'''
    if api is None:
        if required:
            raise AcumosError("AcumosSession.push requires that the API for `{}` be provided".format(param))
    else:
        if not api.startswith('https'):
            logger.warning("Provided `{}` API {} does not begin with 'https'. Your password and token are visible in plaintext!".format(param, api))


def _push_model(dump_dir, push_api, auth_api, options, max_tries=2, extra_headers=None):
    '''Pushes a model to the Acumos server'''
    with ExitStack() as stack:
        model = stack.enter_context(open(path_join(dump_dir, 'model.zip'), 'rb'))
        meta = stack.enter_context(open(path_join(dump_dir, 'metadata.json')))
        proto = stack.enter_context(open(path_join(dump_dir, 'model.proto')))

        files = {'model': ('model.zip', model, 'application/zip'),
                 'metadata': ('metadata.json', meta, 'application/json'),
                 'schema': ('model.proto', proto, 'text/plain')}

        # include a license if one is provided
        if options.license is not None:
            _add_license(dump_dir, options.license)
            license = stack.enter_context(open(path_join(dump_dir, _LICENSE_NAME)))
            files['license'] = (_LICENSE_NAME, license, 'application/json')

        tries = 1
        _post_model(files, push_api, auth_api, tries, max_tries, extra_headers, options)


def _add_license(rootdir, license_str):
    '''Adds a license file to the model root directory'''
    license_dst = path_join(rootdir, _LICENSE_NAME)
    if isfile(license_str):
        shutil.copy(license_str, license_dst)
    else:
        license_dict = {'license': license_str}  # the license team hasn't defined a license schema yet
        dump_artifact(license_dst, data=license_dict, module=json, mode='w')


def _post_model(files, push_api, auth_api, tries, max_tries, extra_headers, options):
    '''Attempts to post the model to Acumos'''
    headers = {'Authorization': get_jwt(auth_api),
               'isCreateMicroservice': 'true' if options.create_microservice else 'false'}
    if extra_headers is not None:
        headers.update(extra_headers)

    resp = requests.post(push_api, files=files, headers=headers)
    if resp.ok:
        logger.info("Model pushed successfully to {}".format(push_api))
        if options.create_microservice:
            try:
                docker_image_uri = resp.json()["dockerImageUri"]
                logger.info(f"Acumos model docker image successfully created: {docker_image_uri}")
            except KeyError:
                logger.warning("Docker image URI could not be found in server response, "
                               "on-boarding server is probably running a version prior to Demeter.")
    else:
        clear_jwt()
        if resp.status_code == 401 and tries != max_tries:
            logger.warning('Model push failed due to an authorization failure. Clearing credentials and trying again')
            _post_model(files, push_api, auth_api, tries + 1, max_tries, extra_headers, options)
        else:
            raise AcumosError("Model push failed: {}".format(_ServerResponse(resp.status_code, resp.reason, resp.text)))


@contextmanager
def _dump_model(model, name, requirements=None):
    '''Generates model artifacts and serializes the model'''
    requirements = Requirements() if requirements is None else requirements

    with TemporaryDirectory() as rootdir:

        model_dir = path_join(rootdir, 'model')
        mkdir(model_dir)

        with AcumosContextManager(model_dir) as context:

            with open(context.build_path('model.pkl'), 'wb') as f:
                dump_model(model, f)

            # generate protobuf definition
            proto_pkg = context.parameters['protobuf_package'] = _random_string()
            protostr = model2proto(model, proto_pkg)
            dump_artifact(rootdir, 'model.proto', data=protostr, module=None, mode='w')

            # generate protobuf source code
            module_name = 'model'
            proto_dir = context.create_subdir('scripts', 'acumos_gen', proto_pkg)
            compile_protostr(protostr, proto_pkg, module_name, proto_dir)

            # generate model metadata
            requirements.reqs.update(context.package_names)
            metadata = create_model_meta(model, name, requirements)
            dump_artifact(rootdir, 'metadata.json', data=metadata, module=json, mode='w')

            # bundle user-provided code
            code_dir = context.create_subdir('scripts', 'user_provided')
            Path(code_dir, '.keep').touch()  # may resolve pruning issues when unzipping

            # copy packages and modules
            pkg_scripts = _gather_package_scripts(requirements.packages)
            _copy_package_scripts(context, pkg_scripts, code_dir)

            scripts = set(_gather_scripts(context, requirements))
            _copy_scripts(scripts, code_dir)

        shutil.make_archive(model_dir, 'zip', model_dir)  # create zip at same level as parent
        shutil.rmtree(model_dir)  # clean up model directory

        yield rootdir


def _copy_dir(src_dir, outdir, name):
    '''Copies a directory to a new location'''
    dst_path = path_join(outdir, name)
    if isdir(dst_path):
        raise AcumosError("Model {} has already been dumped".format(dst_path))
    shutil.copytree(src_dir, dst_path)


def _copy_package_scripts(context, scripts, code_dir):
    '''Moves all gathered package scripts to the context code directory'''
    for script_relpath, script_abspath in scripts:
        script_dirname = dirname(script_relpath)
        context_absdir = context.create_subdir(code_dir, script_dirname, exist_ok=True)
        shutil.copy(script_abspath, context_absdir)


def _gather_package_scripts(packages):
    '''Yields (relpath, abspath) tuples of Python scripts from a sequence of packages'''
    for path in packages:
        path = expanduser(path)
        if not isdir(path):
            raise AcumosError("Path {} is not a directory".format(path))

        for root, dirnames, filenames in walk(path):
            for filename in fnmatch.filter(filenames, '*.py'):
                script_abspath = path_join(root, filename)
                script_relpath = path_join(basename(path), relpath(script_abspath, path))
                yield script_relpath, script_abspath


def _gather_scripts(context, reqs):
    '''Yields absolute paths of Python script dependencies'''
    for script in context.scripts:
        yield script.__file__

    for script_path in reqs.scripts:
        script_abspath = abspath(expanduser(script_path))

        if not exists(script_abspath):
            raise AcumosError("Provided script requirement {} does not exist".format(script_path))

        if isdir(script_abspath):
            globbed_scripts = glob(path_join(script_abspath, _PYGLOB))
            if not globbed_scripts:
                raise AcumosError("Provided script requirement directory {} does not contain Python scripts".format(script_path))
            else:
                yield from globbed_scripts
        elif isfile(script_abspath) and script_abspath.endswith(_PYEXT):
            yield script_abspath
        else:
            raise AcumosError("Provided script requirement {} is invalid. See acumos.metadata.Requirements for documentation".format(script_path))


def _copy_scripts(script_paths, code_dir):
    '''Copies individual scripts to a user-provided code directory'''
    for path in script_paths:
        shutil.copy(path, code_dir)


def _random_string(length=32):
    '''Returns a random string containing ascii characters'''
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

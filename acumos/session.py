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
import contextlib
import requests
import fnmatch
from tempfile import TemporaryDirectory
from os import walk, mkdir
from os.path import dirname, isdir, expanduser, relpath, basename, join as path_join
from pathlib import Path

from acumos.pickler import AcumosContextManager, dump_model
from acumos.metadata import create_model_meta, Requirements
from acumos.utils import dump_artifact, get_qualname
from acumos.exc import AcumosError
from acumos.protogen import model2proto, compile_protostr
from acumos.logging import get_logger
from acumos.modeling import Model
from acumos.auth import get_jwt, clear_jwt


logger = get_logger(__name__)


class AcumosSession(object):
    '''
    A session that enables on-boarding models to Acumos

    Parameters
    ----------
    push_api : str
        The full URL to the Acumos on-boarding server upload API
    auth_api : str
        The full URL to the Acumos on-boarding server authentication API
    '''

    def __init__(self, push_api=None, auth_api=None):
        self.push_api = push_api
        self.auth_api = auth_api

    def push(self, model, name, requirements=None):
        '''
        Pushes a model to Acumos

        Parameters
        ----------
        model : acumos.modeling.Model
            A Acumos model instance
        name : str
            The name of your model
        requirements : acumos.session.Requirements
            Additional Python dependencies that you can optionally specify
        '''
        _assert_valid_input(model, requirements)
        _assert_valid_apis(push_api=self.push_api, auth_api=self.auth_api)

        with _dump_model(model, name, requirements) as dump_dir:
            _push_model(dump_dir, self.push_api, self.auth_api)

    def dump(self, model, name, outdir, requirements=None):
        '''
        Creates a Acumos model .zip on the local file system

        Parameters
        ----------
        model : acumos.modeling.Model
            A Acumos model instance
        name : str
            The name of your model
        outdir : str
            The directory or folder to save your model .zip to
        requirements : acumos.session.Requirements
            Additional Python dependencies that you can optionally specify
        '''
        _assert_valid_input(model, requirements)
        with _dump_model(model, name, requirements) as dump_dir:
            _copy_dir(dump_dir, expanduser(outdir), name)


def _assert_valid_input(model, requirements):
    '''Raises AcumosError if inputs are invalid'''
    if not isinstance(model, Model):
        raise AcumosError("Input `model` must be of type {}".format(get_qualname(Model)))

    if requirements is not None and not isinstance(requirements, Requirements):
        raise AcumosError("Input `requirements` must be of type {}".format(get_qualname(Requirements)))


def _assert_valid_apis(**apis):
    '''Raises AcumosError if api are invalid'''
    for param, api in apis.items():
        if api is None:
            raise AcumosError("An API for `{}` must be provided".format(param))

        if not api.startswith('https'):
            logger.warning("Provided `{}` API {} does not begin with 'https'. Your password and token are visible in plaintext!".format(param, api))


def _push_model(dump_dir, push_api, auth_api, max_tries=2):
    '''Pushes a model to the acumos server'''
    with open(path_join(dump_dir, 'model.zip'), 'rb') as model, \
            open(path_join(dump_dir, 'metadata.json')) as meta, \
            open(path_join(dump_dir, 'model.proto')) as proto:

        files = {'model': ('model.zip', model, 'application/zip'),
                 'metadata': ('metadata.json', meta, 'application/json'),
                 'schema': ('model.proto', proto, 'application/text')}

        _post_model(files, push_api, auth_api, 1, max_tries)


def _post_model(files, push_api, auth_api, tries, max_tries):
    '''Attempts to post the model to Acumos'''
    headers = {'Authorization': get_jwt(auth_api)}
    r = requests.post(push_api, files=files, headers=headers)

    if r.status_code == 201:
        logger.info("Model successfully pushed to {}".format(push_api))
    elif r.status_code == 401:
        clear_jwt()
        if tries == max_tries:
            raise AcumosError("Authentication succeeded but authorization failed: {}".format(r.text))
        else:
            logger.warning('Authorization failed. Trying again')
            _post_model(files, push_api, auth_api, tries + 1, max_tries)
    else:
        raise AcumosError("Model upload failed: {}".format(r.text))


@contextlib.contextmanager
def _dump_model(model, name, requirements=None):
    '''Generates model artifacts and serializes the model'''
    requirements = Requirements() if requirements is None else requirements

    with TemporaryDirectory() as rootdir:

        model_dir = path_join(rootdir, 'model')
        mkdir(model_dir)

        with AcumosContextManager(model_dir) as c:

            with open(c.build_path('model.pkl'), 'wb') as f:
                dump_model(model, f)

            # generate protobuf definition
            proto_pkg = c.parameters['protobuf_package'] = _random_string()
            protostr = model2proto(model, proto_pkg)
            dump_artifact(rootdir, 'model.proto', data=protostr, module=None, mode='w')

            # generate protobuf source code
            module_name = 'model'
            proto_dir = c.create_subdir(path_join('scripts', 'acumos_gen', proto_pkg))
            compile_protostr(protostr, proto_pkg, module_name, proto_dir)

            # generate model metadata
            requirements.reqs.update(c.modules)
            metadata = create_model_meta(model, name, requirements)
            dump_artifact(rootdir, 'metadata.json', data=metadata, module=json, mode='w')

            # bundle user-provided scripts
            user_provided = c.create_subdir(path_join('scripts', 'user_provided'))
            Path(user_provided, '.keep').touch()  # may resolve pruning issues when unzipping

            scripts = _gather_scripts(requirements.packages)
            _copy_scripts(c, scripts)

        shutil.make_archive(model_dir, 'zip', model_dir)  # create zip at same level as parent
        shutil.rmtree(model_dir)  # clean up model directory

        yield rootdir


def _copy_dir(src_dir, outdir, name):
    '''Copies a directory to a new location'''
    dst_path = path_join(outdir, name)
    if isdir(dst_path):
        raise AcumosError("Model {} has already been dumped".format(dst_path))
    shutil.copytree(src_dir, dst_path)


def _copy_scripts(context, scripts, prefix=path_join('scripts', 'user_provided')):
    '''Moves all gathered scripts to the context directory'''
    for script_relpath, script_abspath in scripts:
        context_subdir = path_join(prefix, dirname(script_relpath))
        context_absdir = context.create_subdir(context_subdir, exist_ok=True)
        shutil.copy(script_abspath, context_absdir)


def _gather_scripts(packages):
    '''Yields (relpath, abspath) tuples of Python from a sequence of directories'''
    for path in packages:
        path = expanduser(path)
        if not isdir(path):
            raise AcumosError("Path {} is not a directory".format(path))

        for root, dirnames, filenames in walk(path):
            for filename in fnmatch.filter(filenames, '*.py'):
                script_abspath = path_join(root, filename)
                script_relpath = path_join(basename(path), relpath(script_abspath, path))
                yield script_relpath, script_abspath


def _random_string(length=32):
    '''Returns a random string containing ascii characters'''
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

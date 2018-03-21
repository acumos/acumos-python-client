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
Provides authentication utilities
"""
import json
from os import makedirs, environ
from os.path import extsep, isfile, join as path_join
from getpass import getpass

import requests
from appdirs import user_data_dir
from filelock import FileLock

import acumos
from acumos.exc import AcumosError
from acumos.logging import get_logger
from acumos.utils import load_artifact, dump_artifact


_CONFIG_DIR = user_data_dir('acumos')
_CONFIG_PATH = path_join(_CONFIG_DIR, extsep.join(('config', 'json')))
_LOCK_PATH = path_join(_CONFIG_DIR, extsep.join(('config', 'lock')))

_USERNAME_VAR = 'ACUMOS_USERNAME'
_PASSWORD_VAR = 'ACUMOS_PASSWORD'
_TOKEN_VAR = 'ACUMOS_TOKEN'

logger = get_logger(__name__)

getuser = input


def get_jwt(auth_api):
    '''Returns the jwt string from config or authentication'''
    if _TOKEN_VAR in environ:
        jwt = environ[_TOKEN_VAR]
    else:
        config = _configuration()
        jwt = config.get('jwt')
        if jwt is None:
            jwt = _authenticate(auth_api)
            _configuration(jwt=jwt)
    return jwt


def _authenticate(auth_api):
    '''Authenticates and returns the jwt string'''
    username = environ[_USERNAME_VAR] if _USERNAME_VAR in environ else getuser('Enter username: ')
    password = environ[_PASSWORD_VAR] if _PASSWORD_VAR in environ else getpass('Enter password: ')

    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    request_body = {'request_body': {'username': username, 'password': password}}
    r = requests.post(auth_api, json=request_body, headers=headers)

    if r.status_code != 200:
        raise AcumosError("Authentication failure: {}".format(r.text))

    jwt = r.json()['jwtToken']
    return jwt


def clear_jwt():
    '''Clears the jwt from config'''
    _configuration(jwt=None)


def _configuration(**kwargs):
    '''Optionally updates and returns the config dict'''
    makedirs(_CONFIG_DIR, exist_ok=True)
    lock = FileLock(_LOCK_PATH)

    with lock:
        config = dict() if not isfile(_CONFIG_PATH) else load_artifact(_CONFIG_PATH, module=json, mode='r')

    config.update(kwargs)
    config['version'] = acumos.__version__

    with lock:
        dump_artifact(_CONFIG_PATH, data=config, module=json, mode='w')

    return config

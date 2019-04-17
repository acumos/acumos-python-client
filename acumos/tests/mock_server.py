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
Provides a mock web server
"""
import os
import pexpect
import socket
from collections import namedtuple

import requests
from requests import ConnectionError

from utils import TEST_DIR


_EXPECT_RE = r'.*Running on (?P<server>http://.*:\d+).*'


def _find_port():
    '''Returns an open port number'''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _Config(namedtuple('_Config', ['model_url', 'auth_url', 'ui_url', 'port'])):

    def __new__(cls, port):
        base_url = "http://localhost:{}/v2".format(port)
        ui_url = "{}/ui".format(base_url)
        model_url = "{}/models".format(base_url)
        auth_url = "{}/auth".format(base_url)
        return super().__new__(cls, model_url, auth_url, ui_url, port)


class MockServer(object):

    def __init__(self, timeout=5, use_localhost=True, server=None, extra_envs=None, port=None, https=False):
        '''Creates a test server running with a mock upload API in another process'''
        self._use_localhost = use_localhost
        self._timeout = timeout
        self._child = None
        self._https = https
        self.server = server

        port = _find_port() if port is None else port
        self.config = _Config(port)

        app_path = os.path.join(TEST_DIR, 'connexion_server.py')

        cmd = ['python', app_path, '--port', port]
        if https:
            cmd.append('--https')
        self._cmd = ' '.join(map(str, cmd))

    def __enter__(self):
        '''Spawns the child process and waits for server to start until `timeout`'''
        assert not _server_running(self.config.ui_url), 'A mock server is already running'

        self._child = pexpect.spawn(self._cmd, env=os.environ)
        self._child.expect(_EXPECT_RE, timeout=self._timeout)
        server = self._child.match.groupdict()['server'].decode()
        if self._use_localhost:
            server = server.replace('127.0.0.1', 'localhost')
        self.server = server
        return self

    def __exit__(self, type, value, tb):
        '''Interrupts the server and cleans up'''
        if self._child is not None:
            self._child.sendintr()

    def api(self, route, route_prefix='', readline=True):
        '''Returns the full url with server prepended'''
        cmd = "{}{}{}".format(self.server, route_prefix, route)
        if self._child is not None and self._child.isalive() and readline:
            self._child.readline()  # need to read lines so that the child buffer does not get filled
        return cmd


def _server_running(ui_url):
    '''Returns False if test server is not available'''
    try:
        requests.get(ui_url)
    except ConnectionError:
        return False
    else:
        return True

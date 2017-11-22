# -*- coding: utf-8 -*-
"""
Provides a mock web server
"""
import os
import pexpect

import requests
from requests import ConnectionError

from utils import get_workspace


_PORT = 8887
_BASE_URI = "http://localhost:{}/v2".format(_PORT)
_UI_URI = "{}/ui".format(_BASE_URI)
_EXPECT_RE = r'.*Running on (?P<server>http://.*:\d+).*'

MODEL_URI = "{}/models".format(_BASE_URI)
AUTH_URI = "{}/auth".format(_BASE_URI)


class MockServer(object):

    def __init__(self, timeout=5, use_localhost=True, server=None, extra_envs=None, port=_PORT, https=False):
        '''Creates a test server running with a mock upload API in another process'''
        self._use_localhost = use_localhost
        self._timeout = timeout
        self._child = None
        self._https = https
        self.server = server

        workspace = get_workspace()
        app_path = os.path.join(workspace, 'testing', 'upload', 'app.py')

        cmd = ['python', app_path, '--port', port]
        if https:
            cmd.append('--https')
        self._cmd = ' '.join(map(str, cmd))

    def __enter__(self):
        '''Spawns the child process and waits for server to start until `timeout`'''
        assert not _server_running(), 'A mock server is already running'

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


def _server_running():
    '''Returns False if test server is not available'''
    try:
        requests.get(_UI_URI)
    except ConnectionError:
        return False
    else:
        return True

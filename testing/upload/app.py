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
'''
Provides mock server for unit testing
'''
import argparse

import connexion
from flask import request


TOKEN = 'secrettoken'
USERS = {'foo': 'bar'}


def upload(model, metadata, schema):
    '''Mock upload endpoint'''
    test_header = request.headers.get('X-Test-Header')  # made up header to test extra_headers feature
    jwt = request.headers.get('Authorization')
    if not any(token == TOKEN for token in (jwt, test_header)):
        return {'status': 'Unauthorized'}, 401
    else:
        print("Received model: {}".format(model))
        print("Received metadata: {}".format(metadata))
        print("Received schema: {}".format(schema))
        return {'status': 'OK'}, 201


def authenticate(auth_request):
    '''Mock authentication endpoint'''
    username = auth_request['request_body']['username']
    password = auth_request['request_body']['password']

    if USERS.get(username) == password:
        return {'jwtToken': TOKEN}, 200
    else:
        return {'jwtToken': None, 'resultCode': 401}, 401


if __name__ == '__main__':
    '''Main'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8887)
    parser.add_argument("--https", action='store_true')
    pargs = parser.parse_args()

    app = connexion.App(__name__)
    app.add_api('swagger.yaml')

    if pargs.https:
        app.run(host='localhost', port=pargs.port, ssl_context='adhoc')
    else:
        app.run(host='localhost', port=pargs.port)

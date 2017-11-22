# -*- coding: utf-8 -*-
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
    jwt = request.headers.get('Authorization')
    if jwt != TOKEN:
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

#!/usr/bin/env python
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
# -*- coding: utf-8 -*-
'''
Provides a model runner application that subscribes to iris DataFrame messages and publishes Prediction messages
'''
import argparse
import json
from functools import partial
from os import path

import requests
from flask import Flask, request, make_response, current_app
from google.protobuf import json_format
from gunicorn.app.base import BaseApplication

from acumos.wrapped import load_model


def invoke_method(model_method, downstream):
    '''Consumes and produces protobuf binary data'''
    app = current_app
    content_type = "text/plain;charset=UTF-8"
    bytes_in = request.data
    if not bytes_in:
        if request.form:
            bytes_in = dict(request.form)
        elif request.args:
            bytes_in = dict(request.args)
    if type(bytes_in) == dict:  # attempt to push arguments into JSON for more tolerant parsing
        bytes_in = json.dumps(bytes_in)
    bytes_out = None

    try:
        if app.json_io:
            msg_in = json_format.Parse(bytes_in, model_method.pb_input_type())  # attempt to decode JSON
            msg_out = model_method.from_pb_msg(msg_in)
        else:
            msg_out = model_method.from_pb_bytes(bytes_in)  # default from-bytes method
        if app.json_io:
            bytes_out = json_format.MessageToJson(msg_out.as_pb_msg())
            content_type = "application/javascript;charset=UTF-8"
        else:
            bytes_out = msg_out.as_pb_bytes()
    except json_format.ParseError as e:
        type_input = list(model_method.pb_input_type.DESCRIPTOR.fields_by_name.keys())
        str_reply = "[invoke_method]: Value specification error, expected  {}, {}".format(type_input, e)
        print(str_reply)
        resp = make_response(str_reply, 400)
    except (ValueError, TypeError) as e:
        str_reply = "[invoke_method]: Value conversion error: {}".format(e)
        print(str_reply)
        resp = make_response(str_reply, 400)

    if bytes_out is not None:
        resp = None
        for url in downstream:
            try:
                req_response = requests.post(url, data=bytes_out)
                if resp is None:  # save only first response from downstream list
                    resp = make_response(req_response.content, req_response.status_code)
                    for header_test in ['Content-Type', 'content-type']:  # test for content type to copy from downstream
                        if header_test in req_response:
                            content_type = req_response[header_test]
            except Exception as e:
                print("Failed to publish to downstream url {} : {}".format(url, e))
        if app.return_output:
            if resp is None:  # only if not received from downstream
                resp = make_response(bytes_out, 201)
        else:
            resp = make_response('OK', 201)

    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-Type'] = content_type
    return resp


class StandaloneApplication(BaseApplication):
    '''Custom gunicorn app. Modified from http://docs.gunicorn.org/en/stable/custom.html'''

    def __init__(self, pargs):
        self.parsed_args = pargs
        self.options = {'bind': "{}:{}".format(pargs.host, pargs.port), 'workers': pargs.workers, 'timeout': pargs.timeout}
        super().__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in self.options.items()
                       if key in self.cfg.settings and value is not None])
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return build_app(self.parsed_args)


def build_app(pargs):
    '''Builds and returns a Flask app'''
    downstream = []
    if path.exists('runtime.json'):
        with open('runtime.json') as f:
            runtime = json.load(f)  # ad-hoc way of giving the app runtime parameters
            if not pargs.no_downstream:
                downstream = runtime['downstream']  # list of IP:port/path urls
                print("Found downstream forward routes {}".format(downstream))
    else:
        pargs.return_output = True

    model = load_model(pargs.modeldir)  # refers to ./model dir in pwd. generated by helper script also in this dir

    app = Flask(__name__)
    app.json_io = pargs.json_io  # store io flag
    app.return_output = not pargs.no_output  # store output

    # dynamically add handlers depending on model capabilities
    for method_name, method in model.methods.items():

        handler = partial(invoke_method, model_method=method, downstream=downstream)
        url = "/{}".format(method_name)
        app.add_url_rule(url, method_name, handler, methods=['POST', 'GET'])

        # render down the input in few forms
        typeInput = list(method.pb_input_type.DESCRIPTOR.fields_by_name.keys())

        # render down the output in few forms
        typeOutput = list(method.pb_output_type.DESCRIPTOR.fields_by_name.keys())

        print("Adding route {} [input:{}, output:{}]".format(url, typeInput, typeOutput))

    return app


if __name__ == '__main__':
    '''Main'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=3330)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--modeldir", type=str, default='model', help='specify the model directory to load')
    parser.add_argument("--json_io", action='store_true', help='input+output rich JSON instead of protobuf')
    parser.add_argument("--no_output", action='store_true', help='do not return output in response, only send downstream')
    parser.add_argument("--no_downstream", action='store_true', help='ignore downstream arguments even if in runtime')

    pargs = parser.parse_args()

    StandaloneApplication(pargs).run()

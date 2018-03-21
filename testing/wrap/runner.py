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

from acumos.wrapped import load_model


def invoke_method(model_method, downstream):
    '''Consumes and produces protobuf binary data'''
    app = current_app
    # print("[{:}] JSON I/O Flag: {:}".format(model_method, app.json_io))
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
        str_reply = "[invoke_method]: Value specification error, expected  {:}, {:}".format(type_input, e)
        print(str_reply)
        resp = make_response(str_reply, 400)
    except (ValueError, TypeError) as e:
        str_reply = "[invoke_method]: Value conversion error: {:}".format(e)
        print(str_reply)
        resp = make_response(str_reply, 400)

    if bytes_out is not None:
        has_downstream = False
        for url in downstream:
            try:
                r = requests.post(url, data=bytes_out)
                if not has_downstream:
                    if app.return_output:
                        resp = make_response(r.content, 201)
                    else:
                        resp = make_response('OK', 201)
                    has_downstream = True
            except Exception as e:
                print("Failed to publish to downstream url {} : {}".format(url, e))
        if not has_downstream:
            if app.return_output:
                resp = make_response(bytes_out, 201)
            else:
                resp = make_response('OK', 201)
        # with open('protobuf.out.bin', 'wb') as f:
        #   f.write(msg_out.as_pb_bytes())
        # print(json_format.MessageToJson(msg_out.as_pb_msg()))

    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-Type'] = content_type
    return resp


if __name__ == '__main__':
    '''Main'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3330)
    parser.add_argument("--modeldir", type=str, default='model', help='specify the model directory to load')
    parser.add_argument("--json_io", action='store_true', help='input+output rich JSON instead of protobuf')
    parser.add_argument("--no_output", action='store_true', help='do not return output in response, only send downstream')
    parser.add_argument("--no_forward", action='store_true', help='do not parse a rnutime.json forwarding session')
    parser.add_argument("--host", type=str, default='0.0.0.0', help='specify specific host/IP for server')
    pargs = parser.parse_args()

    downstream = []
    app = Flask(__name__)

    app.return_output = not pargs.no_output  # store output
    if path.exists('runtime.json') and not pargs.no_forward:
        with open('runtime.json') as f:
            runtime = json.load(f)  # ad-hoc way of giving the app runtime parameters
            downstream = runtime['downstream']  # list of IP:port/path urls

    model = load_model(pargs.modeldir)  # refers to ./model dir in pwd. generated by helper script also in this dir

    app.json_io = pargs.json_io  # store io flag

    # dynamically add handlers depending on model capabilities
    for method_name, method in model.methods.items():

        handler = partial(invoke_method, model_method=method, downstream=downstream)
        url = "/{}".format(method_name)
        app.add_url_rule(url, method_name, handler, methods=['POST', 'GET'])

        # render down the input in few forms
        typeInput = list(method.pb_input_type.DESCRIPTOR.fields_by_name.keys())
        msgInput = method.pb_input_type()
        jsonInput = json_format.MessageToDict(msgInput)

        # render down the output in few forms
        typeOutput = list(method.pb_output_type.DESCRIPTOR.fields_by_name.keys())
        msgOutput = method.pb_output_type()
        jsonOutput = json_format.MessageToDict(msgOutput)

        print("Adding route {} [input:{:}, output:{:}]".format(url, typeInput, typeOutput))

    print("Running Flask server on port {}".format(pargs.port))
    app.run(port=pargs.port, host=pargs.host)

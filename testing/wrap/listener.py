#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Provides a "listener" application that listens for Prediction messages
'''
import argparse

from flask import Flask, request

from acumos.wrapped import load_model


if __name__ == '__main__':
    '''Main'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3331)
    parser.add_argument("--modeldir", type=str, default='model')
    pargs = parser.parse_args()

    model = load_model(pargs.modeldir)
    Prediction = model.transform.pb_output_type  # need the Prediction message definition to deserialize

    app = Flask(__name__)

    @app.route('/listen', methods=['POST'])
    def listen():
        bytes_in = request.data
        msg = Prediction.FromString(bytes_in)
        print("Received Prediction message: {}".format(msg.predictions))
        return 'OK', 201

    print("Running Flask server on port {:}".format(pargs.port))
    app.run(port=pargs.port)

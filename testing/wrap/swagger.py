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
Provides a quick way to inspect protobuf and generate swagger example
'''
import argparse
from os.path import exists

import pandas as pd

# import yaml  # note special dependency here

from acumos.wrap.load import load_model
from acumos.wrap.dump import dump_keras_model
from google.protobuf.descriptor import FieldDescriptor


_pb_type_lookup = {
    FieldDescriptor.TYPE_DOUBLE: float,
    FieldDescriptor.TYPE_FLOAT: float,
    FieldDescriptor.TYPE_INT64: int,
    FieldDescriptor.TYPE_UINT64: int,
    FieldDescriptor.TYPE_INT32: int,
    FieldDescriptor.TYPE_FIXED64: int,
    FieldDescriptor.TYPE_FIXED32: int,
    FieldDescriptor.TYPE_BOOL: bool,
    FieldDescriptor.TYPE_STRING: str,
    # TYPE_GROUP = 10
    # TYPE_MESSAGE = 11
    # TYPE_BYTES = 12
    FieldDescriptor.TYPE_UINT32: int,
    FieldDescriptor.TYPE_ENUM: int,
    FieldDescriptor.TYPE_SFIXED32: int,
    FieldDescriptor.TYPE_SFIXED64: int,
    FieldDescriptor.TYPE_SINT32: int,
    FieldDescriptor.TYPE_SINT64: int
}


def dump_yaml(list_methods):
    """Dumps a simple dictionary to a yaml file with the following expected format...
        list_methods = [ {'name':'transform_EXAMPLE', 'in':{'x1':float, 'x2':float, ...}, 'out':{'prediction':float}} ... ]
    """
    print(list_methods)

    # TODO: method fill for actual yaml formatting
    # print(yaml.dump(list_methods))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", type=str, default='model')
    pargs = parser.parse_args()

    if not exists(pargs.modeldir):
        from sklearn.datasets import load_iris
        from keras.models import Sequential
        from keras.layers import Dense

        print("No local model directory found, creating a simple model... '{:}'".format(pargs.modeldir))
        iris = load_iris()
        X = iris.data
        y = pd.get_dummies(iris.target).values

        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y)

        dump_keras_model(model, X, pargs.modeldir, name='keras-iris')

    print("Parsing model source directory... '{:}'".format(pargs.modeldir))
    model = load_model(pargs.modeldir)  # refers to ./model dir in pwd. generated by helper script also in this dir

    listMethod = []
    for method_name in model.methods:   # iterate through known methods
        objMethod = getattr(model, method_name)
        dictInput = {field.name: _pb_type_lookup[field.type] for field in objMethod.msg_in.DESCRIPTOR.fields}
        dictOutput = {field.name: _pb_type_lookup[field.type] for field in objMethod.msg_out.DESCRIPTOR.fields}
        listMethod.append({'name': method_name, 'in': dictInput, 'out': dictOutput})
    dump_yaml(listMethod)


if __name__ == '__main__':
    main()

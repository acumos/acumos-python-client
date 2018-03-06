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
"""
Loads a dumped model
"""
import sys
from os.path import dirname, abspath, join as path_join

from acumos.wrapped import load_model


_TEST_DIR = dirname(abspath(__file__))
_CUSTOM_PACKAGE_FILE = path_join(_TEST_DIR, 'custom_package', '__init__.py')


if __name__ == '__main__':
    '''Main'''
    # remove tests/ from python path to make sure model is using its own custom_package
    for path in ('', _TEST_DIR):
        try:
            sys.path.remove(path)
        except ValueError:
            pass

    model_dir = sys.argv[1]
    model = load_model(model_dir)

    custom_package = sys.modules['custom_package']  # should be imported by model, else KeyError
    assert custom_package.__file__ != _CUSTOM_PACKAGE_FILE  # verify tests/custom_package didn't leak

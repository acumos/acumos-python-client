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
Provides logging utilities
"""
import logging

import acumos


_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s : %(message)s'))

_root = logging.getLogger(acumos.__name__)
_root.setLevel(logging.INFO)
_root.handlers = [_handler, ]
_root.propagate = False


def get_logger(name):
    '''Returns a logger object'''
    return logging.getLogger(name)

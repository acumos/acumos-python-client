#!/usr/bin/env python3
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
Provides testing utils
"""
import os
import subprocess

from acumos.exc import AcumosError


def get_workspace():
    '''Returns WORKSPACE environment variable'''
    workspace = os.environ.get('WORKSPACE')
    if workspace is None:
        raise AcumosError('Jenkins WORKSPACE environment variable is required and must point to root acumos-python-client repository directory.')
    return workspace


def run_command(cmd):
    '''Runs a given command and raises AcumosError on process failure'''
    workspace = get_workspace()
    env = {'PYTHONPATH': workspace, 'PATH': os.environ['PATH']}  # workspace includes acumos/, needed by helpers
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, env=env)
    _, stderr = proc.communicate()
    assert proc.returncode == 0, stderr.decode()

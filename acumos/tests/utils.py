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
import subprocess
import os
from os.path import abspath, dirname


TEST_DIR = dirname(abspath(__file__))


def run_command(cmd):
    '''Runs a given command and raises AcumosError on process failure'''
    env = {'PATH': os.environ['PATH']}
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, env=env)
    _, stderr = proc.communicate()
    assert proc.returncode == 0, stderr.decode()

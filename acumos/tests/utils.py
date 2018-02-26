#!/usr/bin/env python3
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

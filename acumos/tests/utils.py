#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides testing utils
"""
import os

from acumos.exc import AcumosError


def get_workspace():
    '''Returns WORKSPACE environment variable'''
    workspace = os.environ.get('WORKSPACE')
    if workspace is None:
        raise AcumosError('Jenkins WORKSPACE environment variable is required and must point to root acumos-python-client repository directory.')
    return workspace

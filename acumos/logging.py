# -*- coding: utf-8 -*-
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

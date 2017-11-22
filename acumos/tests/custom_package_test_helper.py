#!/usr/bin/env python
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

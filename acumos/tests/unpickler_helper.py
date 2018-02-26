#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unpickles a model, given a context directory
"""
import sys

from acumos.pickler import load_model, AcumosContextManager


if __name__ == '__main__':
    '''Main'''
    context_dir = sys.argv[1]

    with AcumosContextManager(context_dir) as c:
        model_path = c.build_path('model.pkl')
        with open(model_path, 'rb') as f:
            model = load_model(f)  # success if this doesn't explode

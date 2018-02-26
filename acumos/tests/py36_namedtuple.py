# -*- coding: utf-8 -*-
"""
Provides NamedTuple types defined using Python 3.6 syntax
"""
from typing import NamedTuple


class Input(NamedTuple):
    x: int
    y: int


class Output(NamedTuple):
    value: int

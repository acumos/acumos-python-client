# -*- coding: utf-8 -*-
"""
Provides an image Acumos model example

This model returns metadata about input images
"""
import io

import PIL

from acumos.modeling import Model, create_namedtuple
from acumos.session import AcumosSession


ImageShape = create_namedtuple('ImageShape', [('width', int), ('height', int)])


def get_format(data: bytes) -> str:
    '''Returns the format of an image'''
    buffer = io.BytesIO(data)
    img = PIL.Image.open(buffer)
    return img.format


def get_shape(data: bytes) -> ImageShape:
    '''Returns the width and height of an image'''
    buffer = io.BytesIO(data)
    img = PIL.Image.open(buffer)
    shape = ImageShape(width=img.width, height=img.height)
    return shape


model = Model(get_format=get_format, get_shape=get_shape)

session = AcumosSession()
session.dump(model, 'image-model', '.')  # creates ./image-model

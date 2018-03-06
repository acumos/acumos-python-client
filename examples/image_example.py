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

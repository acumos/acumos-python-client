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
'''
Dumps an example model for illustrating acumos_model_runner usage
'''

from acumos.session import AcumosSession
from acumos.modeling import Model, List, Dict, new_type

# allow users to specify a "raw" bytes type. no protobuf message is generated here
Image = new_type(bytes, 'Image')

def image_func(image: Image) -> Image:
    '''Return an image'''
    return Image(image)


if __name__ == '__main__':
    '''Main'''
    model = Model(imgae_func=image_func)

    session = AcumosSession()
    session.dump(raw_model, 'example', '.')

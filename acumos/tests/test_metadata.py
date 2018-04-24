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
Provides metadata tests
"""
import pytest

from acumos.metadata import _gather_requirements, _filter_requirements, Requirements


def test_requirements():
    '''Tests usage of requirement utilities'''
    req = Requirements(packages=['~/foo', '~/bar/'])
    assert req.package_names == {'foo', 'bar'}

    reqs = Requirements(reqs=['foo', 'bar', 'baz'], req_map={'foo': 'bing'}, packages=['~/baz'])
    req_set = _filter_requirements(reqs)
    assert req_set == {'bing', 'bar'}

    reqs = Requirements(reqs=['PIL', 'scikit-learn', 'keras', 'baz', 'collections', 'time', 'sys'],
                        req_map={'PIL': 'pillow'}, packages=['~/baz'])
    req_names, _ = zip(*_gather_requirements(reqs))
    assert set(map(str.lower, req_names)) == {'pillow', 'scikit-learn', 'keras'}


if __name__ == '__main__':
    '''Test area'''
    pytest.main([__file__, ])

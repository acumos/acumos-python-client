#!/usr/bin/env python
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
# -*- coding: utf-8 -*-
"""
Helper script to determine next release version
"""
import sys
import re
import contextlib


RELEASE_BRANCH_PATTERN = r'^release-(?P<major>\d+)\.(?P<minor>\d+).*$'
VERSION_PATTERN = r'^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(\.(?P<release_type>[a-z]+)(?P<release_version>\d+))?$'
VERSION_COMPS = ('major', 'minor', 'patch', 'release_type', 'release_version')


def test_logic():
    '''Poor man's unit tests'''
    # release branch version > current version -> set release type and version
    assert next_version('release-0.4', '0.3.9') == '0.4.0.rc1'
    assert next_version('release-0.4', '0.3.9.dev1') == '0.4.0.rc1'

    # current version > release branch version -> bump patch
    assert next_version('release-0.4', '0.4.0') == ''
    assert next_version('release-0.4', '0.4.0.dev1') == ''
    assert next_version('release-0.4', '0.4.1') == ''

    with _assert_raises(SystemExit):
        next_version('release-0.3', '0.4.0')

    # set dev release type if current version is not dev, else bump release version
    assert next_version('develop', '0.3.11') == '0.3.11.dev1'
    assert next_version('develop', '0.4.0.dev1') == ''


@contextlib.contextmanager
def _assert_raises(cls):
    '''Ensures an exception is raised'''
    try:
        yield
        raise Exception('Did not raise')
    except cls:
        pass


def next_version(branch_name, version_str):
    '''Returns a string corresponding to the next release version'''
    if branch_name.startswith('release'):
        return _bump_release_branch(branch_name, version_str)
    elif branch_name == 'develop':
        return _bump_develop_branch(version_str)
    else:
        raise SystemExit("next_version.py is for release and develop branches only, not {}".format(branch_name))


def _bump_release_branch(branch_name, version_str):
    '''Sets or updates the release version based on the release branch pattern'''
    branch_dict = _get_groupdict(RELEASE_BRANCH_PATTERN, branch_name)
    branch_tuple = tuple(map(int, (branch_dict['major'], branch_dict['minor'])))

    current_tuple = tuple(map(int, version_str.split('.')[:2]))

    if branch_tuple > current_tuple:
        branch_ver = "{}.0.rc1".format(".".join(map(str, branch_tuple)))
        return branch_ver  # branch indicates new release type, use instead
    elif branch_tuple == current_tuple:
        return ''  # use bumpversion to increment release type version
    else:
        sys.exit("Branch version {} cannot be lower than current code version {}".format(branch_name, version_str))


def _bump_develop_branch(version_str):
    '''Sets or updates the develop version based on the current version'''
    major, minor, patch, release, release_ver = _get_version_comps(version_str)
    if release == 'dev':
        return ''  # use bumpversion to increment release type version
    else:
        return "{}.{}.{}.dev1".format(major, minor, patch)  # initialize dev release type


def _get_version_comps(version_str):
    '''Returns (major, minor, patch, release, release_ver) tuple'''
    group_dict = _get_groupdict(VERSION_PATTERN, version_str)
    return map(group_dict.__getitem__, VERSION_COMPS)


def _get_groupdict(pattern, string):
    '''Returns the groupdict from the regex match'''
    match = re.match(pattern, string)
    if match is None:
        raise SystemExit("Did not find pattern {} within branch name {}".format(pattern, string))
    return match.groupdict()


if __name__ == '__main__':
    '''Main'''
    # test_logic()

    branch_name, version_str = sys.argv[1], sys.argv[2]
    sys.stdout.write(next_version(branch_name, version_str))

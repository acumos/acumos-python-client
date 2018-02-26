# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages


# extract __version__ from version file. importing will lead to install failures
setup_dir = os.path.dirname(__file__)
with open(os.path.join(setup_dir, 'acumos', '_version.py')) as file:
    globals_dict = dict()
    exec(file.read(), globals_dict)
    __version__ = globals_dict['__version__']


setup(
    name='acumos',
    version=__version__,
    packages=find_packages(),
    author='Paul Triantafyllou',
    author_email='trianta@research.att.com',
    description=('Acumos client library'),
    setup_requires=['pytest-runner'],
    install_requires=['typing',
                      'protobuf',
                      'requests',
                      'numpy',
                      'dill<0.2.8',
                      'appdirs',
                      'filelock'],
    python_requires='>=3.4',
    license='Apache License 2.0',
)

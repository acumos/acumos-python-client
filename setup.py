# -*- coding: utf-8 -*-
from os.path import dirname, abspath, join as path_join
from setuptools import setup, find_packages


# extract __version__ from version file. importing will lead to install failures
SETUP_DIR = abspath(dirname(__file__))
DOCS_DIR = path_join(SETUP_DIR, 'docs')

with open(path_join(SETUP_DIR, 'acumos', '_version.py')) as file:
    globals_dict = dict()
    exec(file.read(), globals_dict)
    __version__ = globals_dict['__version__']


def _long_descr():
    '''Yields the content of documentation files for the long description'''
    for file in ('README.rst', 'tutorial/index.rst', 'release-notes.rst', 'contributing.rst'):
        doc_path = path_join(DOCS_DIR, file)
        with open(doc_path) as f:
            yield f.read()


setup(
    name='acumos',
    version=__version__,
    packages=find_packages(),
    author='Paul Triantafyllou',
    author_email='trianta@research.att.com',
    description='Acumos client library for building and pushing Python models',
    long_description='\n'.join(_long_descr()),
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
    url='https://www.acumos.org/',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
    ],
    keywords='acumos machine learning model modeling artificial intelligence ml ai',
)

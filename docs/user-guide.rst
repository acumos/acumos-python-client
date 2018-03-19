.. ===============LICENSE_START=======================================================
.. Acumos CC-BY-4.0
.. ===================================================================================
.. Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
.. ===================================================================================
.. This Acumos documentation file is distributed by AT&T and Tech Mahindra
.. under the Creative Commons Attribution 4.0 International License (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://creativecommons.org/licenses/by/4.0
..
.. This file is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ===============LICENSE_END=========================================================

========================
Python Client User Guide
========================

This is a client library that allows modelers to push their Python models to
Acumos.

Installation
============

You will need a Python 3.4+ environment in order to install ``acumos``.
You can use `Anaconda <https://www.anaconda.com/download/>`__
(preferred) or `pyenv <https://github.com/pyenv/pyenv>`__ to install and
manage Python environments.

The ``acumos`` package can be installed with pip:

.. code:: bash

    pip install acumos


Protocol Buffers
----------------

The ``acumos`` package uses protocol buffers and **assumes you have
the protobuf compiler ``protoc`` installed**. Please visit the `protobuf
repository <https://github.com/google/protobuf/releases/tag/v3.4.0>`__
and install the appropriate ``protoc`` for your operating system.
Installation is as easy as downloading a binary release and adding it to
your system ``$PATH``. This is a temporary requirement that will be
removed in a future version of ``acumos``.

**Anaconda Users**: You can easily install ``protoc`` from `an Anaconda
package <https://anaconda.org/anaconda/libprotobuf>`__ via:

.. code:: bash

    conda install -c anaconda libprotobuf

Usage
=====
Please see the tutorial at `Acumos Python Client <https://pypi.org/project/acumos/>`_.

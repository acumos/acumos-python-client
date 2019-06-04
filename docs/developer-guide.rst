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

====================================
Acumos Python Client Developer Guide
====================================

|Build Status|

``acumos`` is a client library that allows modelers to push their Python models
to their own Acumos instance.

You can clone the code at gerrit
`https://gerrit.acumos.org/r/#/admin/projects/acumos-python-client<https://gerrit.acumos.org/r/#/admin/projects/acumos-python-client>`__

Testing
=======

We use a combination of ``tox``, ``pytest``, and ``flake8`` to test
``acumos``. Code which is not PEP8 compliant (aside from E501) will be
considered a failing test. You can use tools like ``autopep8`` to
“clean” your code as follows:

.. code:: bash

    $ pip install autopep8
    $ cd acumos-python-client
    $ autopep8 -r --in-place --ignore E501 acumos/ testing/ examples/

Run tox directly:

.. code:: bash

    $ cd acumos-python-client
    $ export WORKSPACE=$(pwd)  # env var normally provided by Jenkins
    $ tox

You can also specify certain tox environments to test:

.. code:: bash

    $ tox -e py36  # only test against Python 3.6
    $ tox -e flake8  # only lint code



Installation
============

You will need a Python 3.5 or 3.6 environment in order to install ``acumos``.
You can use `Anaconda <https://www.anaconda.com/download/>`__
(preferred) or `pyenv <https://github.com/pyenv/pyenv>`__ to install and
manage Python environments.

If you’re new to Python and need an IDE to start developing, we
recommend using `Spyder <https://github.com/spyder-ide/spyder>`__ which
can easily be installed with Anaconda.

The ``acumos`` package can be installed with pip:

.. code:: bash

    pip install acumos


Protocol Buffers
----------------

The ``acumos`` package uses protocol buffers and **assumes you have
the protobuf compiler** ``protoc`` **installed**. Please visit the `protobuf
repository <https://github.com/google/protobuf/releases/tag/v3.4.0>`__
and install the appropriate ``protoc`` for your operating system.
Installation is as easy as downloading a binary release and adding it to
your system ``$PATH``. This is a temporary requirement that will be
removed in a future version of ``acumos``.

**Anaconda Users**: You can easily install ``protoc`` from `an Anaconda
package <https://anaconda.org/anaconda/libprotobuf>`__ via:

.. code:: bash

    conda install -c anaconda libprotobuf

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

Packaging
=========

The RST files in the docs/ directory are used to publish HTML pages to
ReadTheDocs.io and to build the package long description in setup.py.
The symlink from the subdirectory acumos-package to the docs/ directory
is required for the Python packaging tools.  Those tools build a source
distribution from files in the package root, the directory acumos-package.
The MANIFEST.in file directs the tools to pull files from directory docs/,
and the symlink makes it possible because the tools only look within the
package root.

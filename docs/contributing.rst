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

=======================
Contributing Guidelines
=======================

Branching Model
===============

Follow the workflow described in `A successful Git branching
model <http://nvie.com/posts/a-successful-git-branching-model/>`__.

Versioning
==========

Use the `bumpversion <https://github.com/peritus/bumpversion>`__ tool to
bump the ``acumos`` version. See the `bumpversion config
file <../.bumpversion.cfg>`__ for more details. Here are some common use
cases:

::

    bumpversion major  # 0.1.0 --> 1.0.0
    bumpversion minor  # 0.1.0 --> 0.2.0
    bumpversion patch  # 0.1.0 --> 0.1.1
    bumpversion release_version  # 0.1.0.dev1 --> 0.1.0.dev2

Testing
=======

We use a combination of ``tox``, ``pytest``, and ``flake8`` to test
``acumos``. Code which is not PEP8 compliant (aside from E501) will be
considered a failing test. You can use tools like ``autopep8`` to
“clean” your code as follows:

::

    $ pip install autopep8
    $ cd acumos-python-client
    $ autopep8 -r --in-place --ignore E501 acumos/ testing/ examples/

To test locally, run the Jenkins test script from the repository root.
This will fetch and install isolated Python environments using
``pyenv``:

::

    $ cd acumos-python-client
    $ bash testing/test.sh

You can also run tox directly:

::

    $ cd acumos-python-client
    $ export WORKSPACE=$(pwd)  # env var normally provided by Jenkins
    $ tox

You can also specify certain tox environments to test:

::

    $ tox -e py34  # only test against Python 3.4
    $ tox -e flake8  # only lint code

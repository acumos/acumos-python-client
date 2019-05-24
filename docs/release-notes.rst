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

==================================
Acumos Python Client Release Notes
==================================

v0.8.0, 22 April 2019
=====================

-  Enhancements

   - Users may now specify additional options when pushing their Acumos model. See the options section in the tutorial for more information : `ACUMOS-2281 <https://jira.acumos.org/browse/ACUMOS-2281/>`_, `ACUMOS-2270 <https://jira.acumos.org/browse/ACUMOS-2770/>`_
   - ``acumos`` now supports Keras models built with ``tensorflow.keras`` : `ACUMOS-2276 <https://jira.acumos.org/browse/ACUMOS-2748/>`_

-  Support changes

   - ``acumos`` no longer supports Python 3.4 : `ACUMOS-2766 <https://jira.acumos.org/browse/ACUMOS-2766/>`_


v0.7.2, 19 January 2019
=======================

-  Bug fixes

   - The deprecated authentication API is now considered optional
   - A more portable path solution is now used when saving models, to avoid issues with models developed in Windows


v0.7.1, 17 October 2019
=======================

-  Authentication

   - Username and password authentication has been deprecated
   - Users are now interactively prompted for an onboarding token, as opposed to a username and password

v0.7.0, 20 June 2018
====================

-  Requirements

   - Python script dependencies can now be specified using a Requirements object
   - Python script dependencies found during the introspection stage are now included with the model

v0.6.5, 15 June 2018
====================

-  Bug fixes

   - Don't attempt to use an empty auth token (avoids blank strings to be set in environment)

v0.6.4, 31 May 2018
===================

-  Bug fixes

   - The normalized path of the system base prefix is now used for identifying stdlib packages

v0.6.3, 27 April 2018
=====================

-  Bug fixes

   - Improved dependency inspection when using a virtualenv
   - Removed custom packages from model metadata, as it caused image build failures
   - Fixed Python 3.5.2 ordering bug in wrapped model usage

v0.6.2, 6 April 2018
====================

-  TensorFlow

   - Fixed a serialization issue that occurred when using a frozen graph

v0.6.1, 23 March 2018 
=====================

-  Model upload

   - The JWT is now cleared immediately after a failed upload
   - Additional HTTP information is now included in the error message

v0.6.0, 22 March 2018
=====================

-  Authentication token

   -  A new environment variable ``ACUMOS_TOKEN`` can be used to short-circuit
      the authentication process

-  Extra headers

   -  ``AcumosSession.push`` now accepts an optional ``extra_headers`` argument,
      which will allow users and systems to include additional information when
      pushing models to the onboarding server

v0.5.0
======

-  Modeling

   -  Python 3.6 NamedTuple syntax support now tested
   -  User documentation includes example of new NamedTuple syntax

-  Model wrapper

   -  Model wrapper now has APIs for consuming and producing Python
      dicts and JSON strings

-  Protobuf and protoc

   -  An explicit check for protoc is now made, which raises a more
      informative error message
   -  User documentation is more clear about dependence on protoc, and
      provides an easier way to install protoc via Anaconda

-  Keras

   -  The active keras backend is now included as a tracked module
   -  keras_contrib layers are now supported

v0.4.0
======

-  Replaced library-specific onboarding functions with “new-style”
   models

   -  Support for arbitrary Python functions using type hints
   -  Support for custom user-defined types
   -  Support for TensorFlow models
   -  Improved dependency introspection
   -  Improved object serialization mechanisms
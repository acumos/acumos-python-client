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

v0.7.1
======

-  Authentication

   - Username and password authentication has been deprecated
   - Users are now interactively prompted for an onboarding token, as opposed to a username and password

v0.7.0
======

-  Requirements

   - Python script dependencies can now be specified using a Requirements object
   - Python script dependencies found during the introspection stage are now included with the model

v0.6.5
======

-  Bug fixes

   - Don't attempt to use an empty auth token (avoids blank strings to be set in environment)

v0.6.4
======

-  Bug fixes

   - The normalized path of the system base prefix is now used for identifying stdlib packages

v0.6.3
======

-  Bug fixes

   - Improved dependency inspection when using a virtualenv
   - Removed custom packages from model metadata, as it caused image build failures
   - Fixed Python 3.5.2 ordering bug in wrapped model usage

v0.6.2
======

-  TensorFlow

   - Fixed a serialization issue that occurred when using a frozen graph

v0.6.1
======

-  Model upload

   - The JWT is now cleared immediately after a failed upload
   - Additional HTTP information is now included in the error message

v0.6.0
======

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

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

===========================
Python Client Release Notes
===========================

v0.5
====

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

v0.4
====

-  Replaced library-specific onboarding functions with “new-style”
   models

   -  Support for arbitrary Python functions using type hints
   -  Support for custom user-defined types
   -  Support for TensorFlow models
   -  Improved dependency introspection
   -  Improved object serialization mechanisms

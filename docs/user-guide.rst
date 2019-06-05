.. ===============LICENSE_START============================================================
.. Acumos CC-BY-4.0
.. ========================================================================================
.. Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
.. ========================================================================================
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
.. ===============LICENSE_END==============================================================

===============================
Acumos Python Client User Guide
===============================

``acumos`` is a client library that allows modelers to push their Python models to their own Acumos
instance

all the needed information and prerequisites to use ``acumos`` are depicted in Pypi :
`acumos Pypi <https://pypi.org/project/acumos/>`__

Web-onboarding
==============

When using the session.dump(model, 'my-model', '~/') for web-onboarding, you created the following three files : 

metadata.json
model.proto
model.zip

stored in the 'my-model' folder.

Now you have to zip this three file in an only one : 

.. code-block:: bash

   zip archive.zip metadata.json model.proto model.zip

and then drag and drop this archive.zip towards your Acumos portal, in the "ON-BOARDING BY WEB" page. Or
browse the archive.zip file from the "ON-BOARDING BY WEB" page.


On-board model with license
===========================

If you have a license associated with your model, you can on-board it with your model but you must
name it as "license.json".

If the license file extension is not ‘json’ the license on-boarding will not be possible and if the
name is not ‘license’ Acumos will rename your license file as license.json and you will see your
license file named as license.json in the artifacts table. If you upload a new version of your
license after on-boarding, a number revision will be added to the name of your license file like :
“license-2.json”. To help user create the license file expected by Acumos a license editor is
available on the web : `Acumos license editor <https://pypi.org/project/acumos/>`__



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

=============================
Acumos Python Client Tutorial
=============================

This tutorial provides a brief overview of ``acumos`` for creating
Acumos models. The tutorial is meant to be followed linearly, and some
code snippets depend on earlier imports and objects. Full examples are
available in the ``examples/`` directory of the `Acumos Python client repository <https://gerrit.acumos.org/r/gitweb?p=acumos-python-client.git;a=summary>`__.

#.  `Importing Acumos`_
#.  `Creating A Session`_
#.  `A Simple Model`_
#.  `Exporting Models`_
#.  `Defining Types`_
#.  `Using DataFrames with scikit-learn`_
#.  `Declaring Requirements`_
#.  `Declaring Options`_
#.  `Keras and TensorFlow`_
#. `Testing Models`_
#. `More Examples`_

Importing Acumos
================

First import the modeling and session packages:

.. code:: python

    from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe
    from acumos.session import AcumosSession

Creating A Session
==================

An ``AcumosSession`` allows you to export your models to Acumos. You can
either dump a model to disk locally, so that you can upload it via the
Acumos website, or push the model to Acumos directly.

If you’d like to push directly to Acumos, create a session with the ``push_api`` argument:

.. code:: python

    session = AcumosSession(push_api="https://my.acumos.instance.com/push")

See the onboarding page of your Acumos instance website to find the correct
``push_api`` URL to use.

If you’re only interested in dumping a model to disk, arguments aren’t needed:

.. code:: python

    session = AcumosSession()

A Simple Model
==============

Any Python function can be used to define an Acumos model using `Python
type hints <https://docs.python.org/3/library/typing.html>`__.

Let’s first create a simple model that adds two integers together.
Acumos needs to know what the inputs and outputs of your functions are.
We can use the Python type annotation syntax to specify the function
signature.

Below we define a function ``add_numbers`` with ``int`` type parameters
``x`` and ``y``, and an ``int`` return type. We then build an Acumos
model with an ``add`` method.

**Note:** Function
`docstrings <https://www.python.org/dev/peps/pep-0257/>`__ are included
with your model and used for documentation, so be sure to include one!

.. code:: python

    def add_numbers(x: int, y: int) -> int:
        '''Returns the sum of x and y'''
        return x + y

    model = Model(add=add_numbers)

Exporting Models
================

We can now export our model using the ``AcumosSession`` object created
earlier. The ``push`` and ``dump`` APIs are shown below. The ``dump`` method will
save the model to disk so that it can be onboarded via the Acumos website. The
``push`` method pushes the model directly to Acumos.

.. code:: python

    session.push(model, 'my-model')
    session.dump(model, 'my-model', '~/')  # creates ~/my-model

For more information on how to onboard a dumped model via the Acumos website,
see the `web onboarding guide <https://docs.acumos.org/en/latest/submodules/portal-marketplace/docs/user-guides/portal-user/portal/portal-onboarding-intro.html#on-boarding-by-web>`__.

**Note:** Pushing a model to Acumos will prompt you for an onboarding token if
you have not previously provided one. The interactive prompt can be avoided by
exporting the ``ACUMOS_TOKEN`` environment variable, which corresponds to an
authentication token that can be found in your account settings on the Acumos
website.

Defining Types
==============

In this example, we make a model that can read binary images and output
some metadata about them. This model makes use of a custom type
``ImageShape``.

We first create a ``NamedTuple`` type called ``ImageShape``, which is
like an ordinary ``tuple`` but with field accessors. We can then use
``ImageShape`` as the return type of ``get_shape``. Note how
``ImageShape`` can be instantiated as a new object.

.. code:: python

    import io
    import PIL

    ImageShape = create_namedtuple('ImageShape', [('width', int), ('height', int)])

    def get_format(data: bytes) -> str:
        '''Returns the format of an image'''
        buffer = io.BytesIO(data)
        img = PIL.Image.open(buffer)
        return img.format

    def get_shape(data: bytes) -> ImageShape:
        '''Returns the width and height of an image'''
        buffer = io.BytesIO(data)
        img = PIL.Image.open(buffer)
        shape = ImageShape(width=img.width, height=img.height)
        return shape

    model = Model(get_format=get_format, get_shape=get_shape)

**Note:** Starting in Python 3.6, you can alternatively use this simpler
syntax:

.. code:: python

    from acumos.modeling import NamedTuple

    class ImageShape(NamedTuple):
        '''Type representing the shape of an image'''
        width: int
        height: int

Defining Unstructured Types
===========================

The `create_namedtuple` function allows us to create types with structure,
however sometimes it's useful to work with unstructured data, such as plain
text, dictionaries or byte strings. The `new_type` function allows for just
that.

For example, here's a model that takes in unstructured text, and returns the
number of words in the text:

.. code:: python

    from acumos.modeling import new_type

    Text = new_type(str, 'Text')

    def count(text: Text) -> Text:
        '''Counts the number of words in the text'''
        return len(text.split(' '))

By using the `new_type` function, you inform `acumos` that `Text` is
unstructured, and therefore `acumos` will not create any structured types or
messages for the `count` function.
Version 0.9.x of acumos allows only the use of unstructured types in input and output of
the user defined function.

You can use the `new_type` function to create dictionaries or byte string
type unstructured data as shown below.

.. code:: python

   from acumos.modeling import new_type

   Dict = new_type(dict, 'Dict')

   Image = new_type(byte, 'Image')

Using DataFrames with scikit-learn
==================================

In this example, we train a ``RandomForestClassifier`` using
``scikit-learn`` and use it to create an Acumos model.

When making machine learning models, it’s common to use a dataframe data
structure to represent data. To make things easier, ``acumos`` can
create ``NamedTuple`` types directly from ``pandas.DataFrame`` objects.

``NamedTuple`` types created from ``pandas.DataFrame`` objects store
columns as named attributes and preserve column order. Because
``NamedTuple`` types are like ordinary ``tuple`` types, the resulting
object can be iterated over. Thus, iterating over a ``NamedTuple``
dataframe object is the same as iterating over the columns of a
``pandas.DataFrame``. As a consequence, note how ``np.column_stack`` can
be used to create a ``numpy.ndarray`` from the input ``df``.

Finally, the model returns a ``numpy.ndarray`` of ``int`` corresponding
to predicted iris classes. The ``classify_iris`` function represents
this as ``List[int]`` in the signature return.

.. code:: python

    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()
    X = iris.data
    y = iris.target

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    # here, an appropriate NamedTuple type is inferred from a pandas DataFrame
    X_df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    IrisDataFrame = create_dataframe('IrisDataFrame', X_df)

    # ==================================================================================
    # # or equivalently:
    #
    # IrisDataFrame = create_namedtuple('IrisDataFrame', [('sepal_length', List[float]),
    #                                                     ('sepal_width', List[float]),
    #                                                     ('petal_length', List[float]),
    #                                                     ('petal_width', List[float])])
    # ==================================================================================

    def classify_iris(df: IrisDataFrame) -> List[int]:
        '''Returns an array of iris classifications'''
        X = np.column_stack(df)
        return clf.predict(X)

    model = Model(classify=classify_iris)

Check out the ``sklearn`` examples in the examples directory for full
runnable scripts.

Declaring Requirements
======================

If your model depends on another Python script or package that you wrote, you can
declare the dependency via the ``acumos.metadata.Requirements`` class:

.. code:: python

    from acumos.metadata import Requirements

Note that only pure Python is supported at this time.

Custom Scripts
--------------

Custom scripts can be included by giving ``Requirements`` a sequence of paths
to Python scripts, or directories containing Python scripts. For example, if the
model defined in ``model.py`` depended on ``helper1.py``:

::

    model_workspace/
    ├── model.py
    ├── helper1.py
    └── helper2.py

this dependency could be declared like so:

.. code:: python

    from helper1 import do_thing

    def transform(x: int) -> int:
        '''Does the thing'''
        return do_thing(x)

    model = Model(transform=transform)

    reqs = Requirements(scripts=['./helper1.py'])

    # using the AcumosSession created earlier:
    session.push(model, 'my-model', reqs)
    session.dump(model, 'my-model', '~/', reqs)  # creates ~/my-model

Alternatively, all Python scripts within ``model_workspace/`` could be included
using:

.. code:: python

    reqs = Requirements(scripts=['.'])

Custom Packages
---------------

Custom packages can be included by giving ``Requirements`` a sequence of paths to
Python packages, i.e. directories with an ``__init__.py`` file. Assuming that the
package ``~/repos/my_pkg`` contains:

::

    my_pkg/
    ├── __init__.py
    ├── bar.py
    └── foo.py

then you can bundle ``my_pkg`` with your model like so:

.. code:: python

    from my_pkg.bar import do_thing

    def transform(x: int) -> int:
        '''Does the thing'''
        return do_thing(x)

    model = Model(transform=transform)

    reqs = Requirements(packages=['~/repos/my_pkg'])

    # using the AcumosSession created earlier:
    session.push(model, 'my-model', reqs)
    session.dump(model, 'my-model', '~/', reqs)  # creates ~/my-model

Requirement Mapping
-------------------

Python packaging and `PyPI <https://pypi.org/>`__ aren’t
perfect, and sometimes the name of the Python package you import in your
code is different than the package name used to install it. One example
of this is the ``PIL`` package, which is commonly installed using `a fork
called pillow <https://pillow.readthedocs.io>`_ (i.e.
``pip install pillow`` will provide the ``PIL`` package).

To address this inconsistency, the ``Requirements``
class allows you to map Python package names to PyPI package names. When
your model is analyzed for dependencies by ``acumos``, this mapping is
used to ensure the correct PyPI packages will be used.

In the example below, the ``req_map`` parameter is used to declare a
requirements mapping from the ``PIL`` Python package to the ``pillow``
PyPI package:

.. code:: python

    reqs = Requirements(req_map={'PIL': 'pillow'})

Declaring Options
=================

The ``acumos.metadata.Options`` class is a collection of options that users may
wish to specify along with their Acumos model. If an ``Options`` instance is not
provided to ``AcumosSession.push``, then default options are applied. See the
class docstring for more details.

Below, we demonstrate how options can be used to include additional model metadata
and influence the behavior of the Acumos platform. For example, a license can be
included with a model via the ``license`` parameter, either by providing a license
string or a path to a license file. Likewise, we can specify whether or not the Acumos
platform should eagerly build the model microservice via the ``create_microservice``
parameter.

.. code:: python

    from acumos.metadata import Options

    opts = Options(license="Apache 2.0",       # "./path/to/license_file" also works
                   create_microservice=False,  # don't build the microservice yet

    session.push(model, 'my-model', options=opts)

Keras and TensorFlow
====================

Check out the Keras and TensorFlow examples in the ``examples/`` directory of
the `Acumos Python client repository <https://gerrit.acumos.org/r/gitweb?p=acumos-python-client.git;a=summary>`__.

Testing Models
==============

The ``acumos.modeling.Model`` class wraps your custom functions and
produces corresponding input and output types. This section shows how to
access those types for the purpose of testing. For simplicity, we’ll
create a model using the ``add_numbers`` function again:

.. code:: python

    def add_numbers(x: int, y: int) -> int:
        '''Returns the sum of x and y'''
        return x + y

    model = Model(add=add_numbers)

The ``model`` object now has an ``add`` attribute, which acts as a
wrapper around ``add_numbers``. The ``add_numbers`` function can be
invoked like so:

.. code:: python

    result = model.add.inner(1, 2)
    print(result)  # 3

The ``model.add`` object also has a corresponding *wrapped* function
that is generated by ``acumos.modeling.Model``. The wrapped function is
the primary way your model will be used within Acumos.

We can access the ``input_type`` and ``output_type`` attributes to test
that the function works as expected:

.. code:: python

    AddIn = model.add.input_type
    AddOut = model.add.output_type

    add_in = AddIn(1, 2)
    print(add_in)  # AddIn(x=1, y=2)

    add_out = AddOut(3)
    print(add_out)  # AddOut(value=3)

    model.add.wrapped(add_in) == add_out  # True

More Examples
=============

Below are some additional function examples. Note how ``numpy`` types
can even be used in type hints, as shown in the ``numpy_sum`` function.

.. code:: python

    from collections import Counter
    import numpy as np

    def list_sum(x: List[int]) -> int:
        '''Computes the sum of a sequence of integers'''
        return sum(x)

    def numpy_sum(x: List[np.int32]) -> np.int32:
        '''Uses numpy to compute a vectorized sum over x'''
        return np.sum(x)

    def count_strings(x: List[str]) -> Dict[str, int]:
        '''Returns a count mapping from a sequence of strings'''
        return Counter(x)

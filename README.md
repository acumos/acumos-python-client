# acumos-python-client
[![Build Status](http://cognita-dev1-jenkins.eastus.cloudapp.azure.com:8080/job/acumos-python-client/badge/icon)](http://cognita-dev1-jenkins.eastus.cloudapp.azure.com:8080/job/acumos-python-client/)

A client library that allows developers to push their Python models to Acumos.


## Installation
You will need a Python 3.4+ environment in order to install `acumos`. You can use [Anaconda](https://www.anaconda.com/download/) (preferred) or [pyenv](https://github.com/pyenv/pyenv) to install and manage Python environments.

If you're new to Python and need an IDE to start developing, we recommend using [Spyder](https://github.com/spyder-ide/spyder) which can easily be installed with Anaconda.

To install `acumos-python-client` just clone this repository and use pip:

```
git clone <acumos-python-client repo url>
pip install ./acumos-python-client
```

**Note:** This version of `acumos` uses protocol buffers and assumes you have the protobuf compiler `protoc` installed. Please visit the [protobuf repository](https://github.com/google/protobuf/releases/tag/v3.4.0) and install the appropriate `protoc` for your operating system. This is a temporary requirement that will be removed in a future version of `acumos`.

## Tutorial
This tutorial provides a brief overview of `acumos` for creating Acumos models. The tutorial is meant to be followed linearly, and some code snippets depend on earlier imports and objects. Full examples are available in [the examples directory](examples/).

1. [Importing Acumos](#markdown-header-importing-acumos)
2. [Creating A Session](#markdown-header-creating-a-session)
3. [A Simple Model](#markdown-header-a-simple-model)
4. [Exporting Models](#markdown-header-exporting-models)
5. [Defining Types](#markdown-header-defining-types)
6. [Using DataFrames With scikit-learn](#markdown-header-using-dataframes-with-scikit-learn)
7. [Declaring Requirements](#markdown-header-declaring-requirements)
8. [TensorFlow](#markdown-header-tensorflow)
9. [Testing Models](#markdown-header-testing-models)
10. [More Examples](#markdown-header-more-examples)

### Importing Acumos

First import the modeling and session packages:

```
from acumos.modeling import Model, List, Dict, create_namedtuple, create_dataframe
from acumos.session import AcumosSession, Requirements
```

### Creating A Session

An `AcumosSession` allows you to export your models to Acumos. You can either dump a model to disk locally, so that you can upload it via the Acumos GUI, or push the model to Acumos directly.

If you'd like to push to Acumos, create a session with the `push_api` and `auth_api` arguments:

```
# replace these fake APIs with ones appropriate for your instance! 
session = AcumosSession(push_api="https://my.acumos.instance.com/upload",
                        auth_api="https://my.acumos.instance.com/auth")
```

If you're only interested in dumping a model to disk, the API arguments aren't needed:

```
session = AcumosSession()
```

### A Simple Model

Any Python function can be used to define an Acumos model using [Python type hints](https://docs.python.org/3/library/typing.html).

Let's first create a simple model that adds two integers together. Acumos needs to know what the inputs and outputs of your functions are. We can use the Python type annotation syntax to specify the function signature.

Below we define a function `add_numbers` with `int` type parameters `x` and `y`, and an `int` return type. We then build an Acumos model with an `add` method.

**Note:** Function [docstrings](https://www.python.org/dev/peps/pep-0257/) are included with your model and used for documentation, so be sure to include one!

```
def add_numbers(x: int, y: int) -> int:
    '''Returns the sum of x and y'''
    return x + y

model = Model(add=add_numbers)
```

### Exporting Models

We can now export our model using the `AcumosSession` object created earlier. The `push` and `dump` APIs are shown below.

**Note:** Pushing a model to Acumos will prompt you for your username and password. You can also set the `ACUMOS_USERNAME` and `ACUMOS_PASSWORD` environment variables to avoid being prompted.

```
session.push(model, 'my-model')
session.dump(model, 'my-model', '~/')  # creates ~/my-model
```

### Defining Types

In this example, we make a model that can read binary images and output some metadata about them. This model makes use of a custom type `ImageShape`.

We first create a `NamedTuple` type called `ImageShape`, which is like an ordinary `tuple` but with field accessors. We can then use `ImageShape` as the return type of `get_shape`. Note how `ImageShape` can be instantiated as a new object.

```
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
```

### Using DataFrames With scikit-learn

In this example, we train a `RandomForestClassifier` using `scikit-learn` and use it to create an Acumos model.

When making machine learning models, it's common to use a dataframe data structure to represent data. To make things easier, `acumos` can create `NamedTuple` types directly from `pandas.DataFrame` objects.

`NamedTuple` types created from `pandas.DataFrame` objects store columns as named attributes and preserve column order. Because `NamedTuple` types are like ordinary `tuple` types, the resulting object can be iterated over. Thus, iterating over a `NamedTuple` dataframe object is the same as iterating over the columns of a `pandas.DataFrame`. As a consequence, note how `np.column_stack` can be used to create a `numpy.ndarray` from the input `df`.

Finally, the model returns a `numpy.ndarray` of `int` corresponding to predicted iris classes. The `classify_iris` function represents this as `List[int]` in the signature return.

```
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X_df = pd.DataFrame(X, columns=columns)

IrisDataFrame = create_dataframe('IrisDataFrame', X_df)

def classify_iris(df: IrisDataFrame) -> List[int]:
    '''Returns an array of iris classifications'''
    X = np.column_stack(df)
    return clf.predict(X)
    
model = Model(classify=classify_iris)
```


Check out the [`sklearn` examples](examples/sklearn) in the [examples directory](examples) for full runnable scripts.

### Declaring Requirements

If your model depends on another Python package that you wrote, you can declare the package via the `Requirements` class. Note that only pure Python packages are supported at this time.

Assuming that the package `~/repos/my_pkg` contains:

```
my_pkg/
├── __init__.py
├── bar.py
└── foo.py
```

then you can bundle `my_pkg` with your model like so:

```
from my_pkg.bar import do_thing

def transform(x: int) -> int:
    '''Does the thing'''
    return do_thing(x)

model = Model(transform=transform)

reqs = Requirements(packages=['~/repos/my_pkg'])

# using the AcumosSession created earlier:
session.push(model, 'my-model', reqs)
session.dump(model, 'my-model', '~/', reqs)  # creates ~/my-model
```

### TensorFlow

Check out the [TensorFlow example](examples/tensorflow_example.py) in the [examples directory](examples).
 
### Testing Models

The `acumos.modeling.Model` class wraps your custom functions and produces corresponding input and output types. This section shows how to access those types for the purpose of testing. For simplicity, we'll create a model using the `add_numbers` function again:

```
def add_numbers(x: int, y: int) -> int:
    '''Returns the sum of x and y'''
    return x + y

model = Model(add=add_numbers)
```

The `model` object now has an `add` attribute, which acts as a wrapper around `add_numbers`. The `add_numbers` function can be invoked like so:

```
result = model.add.inner(1, 2)
result == 3  # True
```

The `model.add` object also has a corresponding *wrapped* function that is generated by `acumos.modeling.Model`. The wrapped function is the primary way your model will be used within Acumos.

We can access the `input_type` and `output_type` attributes to test that the function works as expected:

```
AddIn = model.add.input_type
AddOut = model.add.output_type

add_in = AddIn(1, 2)
print(add_in)  # AddIn(x=1, y=2)

add_out = AddOut(3)
print(add_out)  # AddOut(value=3)

model.add.wrapped(add_in) == add_out  # True
```

### More Examples

Below are some additional function examples. Note how `numpy` types can even be used in type hints, as shown in the `numpy_sum` function.

```
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
```

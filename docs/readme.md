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

## Usage

See our [tutorial](tutorial.md) to learn how to use `acumos`.

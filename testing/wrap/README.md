# testing/wrap
This directory provides example applications that demonstrate how wrapped models work

# Scripts
## dump\_example\_model.py
This script trains a scikit-learn model on the iris dataset and dumps the model to the present working directory.

```
$ cd testing/wrap
$ python dump_example_model.py
```

This creates additional files that would be used for pushing the dumped model to the Acumos upload server:

```
.
├── model
│   ├── metadata.json
│   ├── model.pkl
│   ├── model.py
│   ├── wrap.json
│   ├── model.proto
│   └── model.zip

```

## talker.py
This script continuously sends DataFrame messages to the model runner script every 5 seconds by default.

## runner.py
This script loads the dumped model and uses it to dynamically add `flask` endpoints. In this instance, the scikit-learn model implements the `transform` API which results in a `/transform` endpoint.

```
usage: runner.py [-h] [--port PORT] [--modeldir MODELDIR] [--json_io]
                 [--return_output]

optional arguments:
  -h, --help           show this help message and exit
  --port PORT
  --modeldir MODELDIR  specify the model directory to load
  --json_io            input+output rich JSON instead of protobuf
  --return_output      return output in response instae of just downstream
```

To test JSON-based endpoints, you can specify the flag `--json_io` and the app will attempt ot decode and encode outputs in JSON.

Note that the downstream applications that are being "published" to are defined in `runtime.json` file via the `downstream` key. However, you can
also request that the output is included in the response with the flag `--return_output`.

The following examples are provided for curl-based evaluation from a command-line.

```
(as GET)
curl -X GET "http://localhost:3330/transform?x0=123&x2=0.31&x1=0.77&x3=0.12"

(as POST)
curl --data x1=123 -d x0=0.2 -d x2=0.5 -d x3=0.1  -X POST http://localhost:3330/transform

(as POST, with multiple)
curl --data x1=123 -d x0=0.2 -d x2=0.5 -d x3=0.1 -d x1=2 -d x0=0.1 -d x2=3 -d x3=0.4  -X POST http://localhost:3330/transform
```


## listen.py
This script receives Prediction messages produced by the model runner script and prints them to console.

# Running the example
Run all three applications together to create the pipeline:

```
$ python talker.py &> /dev/null &
$ python runner.py &> /dev/null &
$ python listener.py 
```

## Running your own example.
To aide in the testing of your own work, each of these scripts have an additional
argument `--modeldir` that can be used to point to your own `model` directory
Additionally, while you are encouraged to derive your own `talker` script, you can
also utilize this script to feed test samples to your script by providing
a CSV-based file.  This changes the run patterns for these scripts as thus.

```
$ python talker.py --modeldir /some/path/model --csvdata /some/path/data.csv &> /dev/null &
$ python runner.py --modeldir /some/path/model &> /dev/null &
$ python listener.py --modeldir /some/path/model
```

# Swagger and Wrapper example
To aide in the development and export of models to a swagger/webapp interface
a sample script was created to inspect models and generate python `dict` wrapper.  To call
this sample jut point it at your target model directory and a simple output will be
generated for all methods.  If you don't have a model a simple model will be dumped to the
target directory.

```
$ python swagger.py --modeldir /some/path/model 

(output)
[{
    'name': 'transform', 
    'out': {
        'predictions': <class 'int'>
    }, 
    'in': {
        'x1': <class 'float'>, 
        'x3': <class 'float'>, 
        'x0': <class 'float'>, 
        'x2': <class 'float'>
    }
}]
```

# -*- coding: utf-8 -*-
# ===============LICENSE_START=======================================================
# Acumos Apache-2.0
# ===================================================================================
# Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
# ===================================================================================
# This Acumos software file is distributed by AT&T and Tech Mahindra
# under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============LICENSE_END=========================================================
"""
Provides custom pickle utilities
"""
import json
import inspect
import contextlib
import tempfile
from os import makedirs
from os.path import basename, isdir, isfile, join as path_join
from copy import deepcopy
from functools import partial
from typing import GenericMeta, Dict, List

import dill

from acumos.modeling import _is_namedtuple, create_namedtuple, Empty
from acumos.exc import AcumosError


_BLACKLIST = {'builtins', '__main__'}
_DEFAULT = 'default'

_contexts = dict()

dump_model = partial(dill.dump, recurse=True)
dumps_model = partial(dill.dumps, recurse=True)
load_model = dill.load
loads_model = dill.loads


def _save_annotation(pickler, obj):
    '''Workaround for dill annotation serialization bug'''
    if obj.__origin__ in (Dict, List):
        # recursively save object
        t = obj.__origin__
        args = obj.__args__
        pickler.save_reduce(_load_annotation, (t, args), obj=obj)
    else:
        # eventually hit base type, then use stock pickling logic. temp revert prevents infinite recursion
        t = obj
        args = None
        with _revert_dispatch(GenericMeta):
            pickler.save_reduce(_load_annotation, (t, args), obj=obj)


def _load_annotation(t, args):
    '''Workaround for dill annotation serialization bug'''
    if t is Dict and args is not None:
        return Dict[args[0], args[1]]
    elif t is List and args is not None:
        return List[args[0]]
    else:
        return t


def _save_namedtuple(pickler, obj):
    '''Workaround for dill NamedTuple serialization bug'''
    pickler.save_reduce(_load_namedtuple, (obj.__name__, obj._field_types), obj=obj)


def _load_namedtuple(name, field_types):
    '''Workaround for dill NamedTuple serialization bug'''
    return create_namedtuple(name, [(k, v) for k, v in field_types.items()])


@contextlib.contextmanager
def _revert_dispatch(t):
    '''Temporarily removes a type from the dispatch table to use existing serialization logic'''
    f = dill.Pickler.dispatch.pop(t)
    try:
        yield
    finally:
        dill.Pickler.dispatch[t] = f


def _save_keras(pickler, obj):
    '''Serializes a keras model to a context directory'''
    from keras.backend import backend

    context = get_context()
    model_subdir_abspath = context.create_subdir()  # /path/to/context/root/abc123

    model_name = 'model.h5'
    model_subpath = path_join(basename(model_subdir_abspath), model_name)  # abc123/model.h5
    model_abspath = path_join(context.abspath, model_subpath)  # /path/to/context/root/abc123/model.h5

    obj.save(model_abspath)
    context.add_module('h5py')  # needed for keras model serialization
    context.add_module(backend())  # adds name of active keras backend

    # check for custom or contrib layer modules
    layer_modules = (_get_top_module(l)[1] for l in obj.layers)
    special_layers = tuple((l.__class__, m) for l, m in zip(obj.layers, layer_modules) if m != 'keras')
    for _, module in special_layers:
        context.add_module(module)

    # store subpath because context root can change, and special layer classes to have them imported before load
    pickler.save_reduce(_load_keras, (model_subpath, special_layers), obj=obj)


def _load_keras(model_subpath, special_classes):
    '''Loads a keras model from a context subdirectory'''
    from keras.models import load_model as load_keras_model

    context = get_context()
    model_path = context.build_path(model_subpath)  # /different/path/to/context/root/abc123/model.h5
    model = load_keras_model(model_path)
    return model


def _save_tf_tensor(pickler, tensor):
    '''Saves a TensorFlow tensor object'''
    import tensorflow as tf
    pickler.save_reduce(_load_tf_tensor, (tensor.name, tensor.graph, tf.get_default_session()), obj=tensor)


def _load_tf_tensor(name, graph, session):
    '''Loads a TensorFlow tensor object'''
    return graph.get_tensor_by_name(name)


def _save_tf_operation(pickler, op):
    '''Saves a TensorFlow operation object'''
    import tensorflow as tf
    pickler.save_reduce(_load_tf_operation, (op.name, op.graph, tf.get_default_session()), obj=op)


def _load_tf_operation(name, graph, session):
    '''Loads a TensorFlow operation object'''
    return graph.get_operation_by_name(name)


def _save_tf_session(pickler, session):
    '''Saves a TensorFlow session'''
    import tensorflow as tf

    context = get_context()
    model_subdir_abspath = context.create_subdir()  # /path/to/context/root/abc123

    model_name = 'model.ckpt'
    model_subpath = path_join(basename(model_subdir_abspath), model_name)  # abc123/model.ckpt
    model_abspath = path_join(context.abspath, model_subpath)  # /path/to/context/root/abc123/model.ckpt

    interactive = isinstance(session, tf.InteractiveSession)

    # graph is included in save_reduce in order to make sure that it is restored before trying
    # to restore the session, which would result in an error due to missing tensors / operations
    graph = session.graph

    saver = tf.train.Saver()
    saver.save(session, model_abspath)
    pickler.save_reduce(_load_tf_session, (model_subpath, interactive, graph), obj=session)


def _load_tf_session(model_subpath, interactive, graph):
    '''Loads a TensorFlow session'''
    import tensorflow as tf

    context = get_context()
    model_path = context.build_path(model_subpath)  # /different/path/to/context/root/abc123/model.ckpt

    with graph.as_default():
        sess = tf.InteractiveSession() if interactive else tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
    return sess


def _save_tf_graph(pickler, graph):
    '''Saves a TensorFlow graph'''
    import tensorflow as tf

    context = get_context()
    model_subdir_abspath = context.create_subdir()  # /path/to/context/root/abc123

    graph_name = 'graph.meta'
    graph_subpath = path_join(basename(model_subdir_abspath), graph_name)
    graph_abspath = path_join(context.abspath, graph_subpath)

    with graph.as_default():
        tf.train.export_meta_graph(graph_abspath)  # exports default graph
    pickler.save_reduce(_load_tf_graph, (graph_subpath, ), obj=graph)


def _load_tf_graph(graph_subpath):
    '''Loads a TensorFlow graph'''
    import tensorflow as tf

    context = get_context()
    graph_path = context.build_path(graph_subpath)

    graph = tf.Graph()
    with graph.as_default():
        tf.train.import_meta_graph(graph_path)
    return graph


dill.Pickler.dispatch[GenericMeta] = _save_annotation


_CUSTOM_DISPATCH = {
    'keras.engine.training.Model': _save_keras,
    'tensorflow.python.framework.ops.Tensor': _save_tf_tensor,
    'tensorflow.python.framework.ops.Operation': _save_tf_operation,
    'tensorflow.python.client.session.BaseSession': _save_tf_session,
    'tensorflow.python.framework.ops.Graph': _save_tf_graph,
}


@contextlib.contextmanager
def _patch_dill():
    '''Temporarily patches the dill Pickler dispatch table to support custom serialization within a context'''
    try:
        dispatch = dill.Pickler.dispatch
        dill.Pickler.dispatch = deepcopy(dispatch)

        pickler_save = dill.Pickler.save

        def wrapped_save(pickler, obj, save_persistent_id=True):
            '''Hook that intercepts objects about to be saved'''
            _catch_object(obj)

            if _is_namedtuple(obj) and obj is not Empty:
                _save_namedtuple(pickler, obj)
            else:
                pickler_save(pickler, obj, save_persistent_id)

        dill.Pickler.save = wrapped_save
        yield
    finally:
        dill.Pickler.dispatch = dispatch
        dill.Pickler.save = pickler_save


def _catch_object(obj):
    '''Inspects object and executes custom serialization / bookkeeping logic'''

    # dynamically extend dispatch table to prevent unnecessary imports / dependencies
    obj_type = obj if inspect.isclass(obj) else type(obj)
    if obj_type not in dill.Pickler.dispatch:
        for path in _get_mro_paths(obj_type):
            if path in _CUSTOM_DISPATCH:
                dill.Pickler.dispatch[obj_type] = _CUSTOM_DISPATCH[path]

    mod, name = _get_top_module(obj)
    if name is not None and name not in _BLACKLIST:
        # this check is the current solution for discarding unwanted scripts like __main__
        if mod.__package__:
            context = get_context()
            context.add_module(name)


def _get_top_module(obj):
    '''Returns the top-level module for a given object'''
    mod = inspect.getmodule(obj)
    if mod is not None:
        base, _, _ = mod.__name__.partition('.')
    else:
        base = None
    return mod, base


def _get_mro_paths(type_):
    '''Yields import path string for each entry in `getmro`'''
    for t in inspect.getmro(type_):
        yield "{}.{}".format(t.__module__, t.__name__)


class AcumosContext(object):
    '''Represents a workspace for a model that is being dumped'''

    def __init__(self, root_dir):
        if not isdir(root_dir):
            raise AcumosError("AcumosContext root directory {} does not exist".format(root_dir))

        self._root_dir = root_dir
        self._modules = {'acumos', 'dill'}
        self._params_path = path_join(root_dir, 'context.json')
        self.parameters = self._load_params()

    def create_subdir(self, name=None, exist_ok=False):
        '''Creates a new directory within the context root and returns the absolute path'''
        if name is None:
            tdir = tempfile.mkdtemp(dir=self._root_dir)
        else:
            tdir = path_join(self._root_dir, name)
            makedirs(tdir, exist_ok=exist_ok)
        return tdir

    def build_path(self, *args):
        '''Returns an absolute path starting from the context root'''
        return path_join(self._root_dir, *args)

    def add_module(self, module):
        '''Adds a module to the context module set'''
        self._modules.add(module)

    @property
    def abspath(self):
        return self._root_dir

    @property
    def basename(self):
        return basename(self._root_dir)

    @property
    def modules(self):
        return set(self._modules)

    def _load_params(self):
        '''Returns a parameters dict'''
        if isfile(self._params_path):
            with open(self._params_path) as f:
                return json.load(f)
        else:
            return dict()

    def save_params(self):
        '''Saves a parameters json file within the context root'''
        with open(self._params_path, 'w') as f:
            json.dump(self.parameters, f)


@contextlib.contextmanager
def AcumosContextManager(rootdir=None, name=_DEFAULT):
    '''Context manager that provides a AcumosContext object'''
    with _patch_dill():
        if name in _contexts:
            raise AcumosError("AcumosContext '{}' has already been created. Use `get_context` to access it.".format(name))
        try:
            with _DirManager(rootdir) as rootdir:
                context = AcumosContext(rootdir)
                _contexts[name] = context
                yield context
                context.save_params()
        finally:
            del _contexts[name]


@contextlib.contextmanager
def _DirManager(dir_=None):
    '''Wrapper that passes dir_ through or creates a temporary directory'''
    if dir_ is not None:
        if not isdir(dir_):
            raise AcumosError("Provided AcumosContext rootdir {} does not exist".format(dir_))
        yield dir_
    else:
        with tempfile.TemporaryDirectory() as tdir:
            yield tdir


def get_context(name=_DEFAULT):
    '''Returns an existing AcumosContext'''
    if name in _contexts:
        return _contexts[name]
    else:
        raise AcumosError("AcumosContext '{}' has not been created".format(name))

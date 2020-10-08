import os
import sys
import typing as T

import pytest

from acumos.session import AcumosSession, Options
from acumos.modeling import Model, NamedTuple, new_type


@pytest.fixture()
def acumos_session() -> AcumosSession:
    return AcumosSession(push_api=os.environ.get("ACUMOS_PUSH_API_URL"),
                         auth_api=os.environ.get("ACUMOS_AUTH_API_URL"))


@pytest.fixture()
def acumos_options() -> Options:
    return Options(create_microservice=False)


@pytest.fixture()
def model_name_prefix() -> str:
    python_version = "".join(map(str, sys.version_info[0:2]))
    return f"py{python_version}-integration-"


Image = new_type(bytes, 'Image', {'dcae_input_name': 'a', 'dcae_output_name': 'a'}, 'example description')
Text = new_type(str, 'Text', {'dcae_input_name': 'a', 'dcae_output_name': 'a'}, 'example description')
Dict = new_type(dict, 'Dict', {'dcae_input_name': 'a', 'dcae_output_name': 'a'}, 'example description')
Coord = NamedTuple('Coord', [("x", int), ("y", int)])

types = [
    pytest.param(int, id="structured-int"),
    pytest.param(Coord, id="structured-nametuple"),
    pytest.param(T.List[int], id="structured-list"),
    pytest.param(T.List[Coord], id="structured-list-of-nametuple"),
    pytest.param(T.Dict[str, int], id="structured-dict"),
    pytest.param(T.Dict[str, Coord], id="structured-dict-of-nametuple"),
    pytest.param(Image, id="unstructured-bytes"),
    pytest.param(Text, id="unstructured-str"),
    pytest.param(Dict, id="unstructured-dict"),
]


@pytest.fixture(params=types)
def input_type(request):
    return request.param


@pytest.fixture(params=types)
def output_type(request):
    return request.param


@pytest.fixture()
def model(input_type, output_type):
    def f(x: input_type) -> output_type:
        pass

    return Model(f=f)


def test_push_generic_model(model, acumos_session, acumos_options, model_name_prefix, request):
    name = request.node.name
    acumos_session.push(model, model_name_prefix + name.split("[")[1][:-1], options=acumos_options)


def count(text: Text) -> int:
    return len(text)


def create_text(x: int, y: int) -> Text:
    return str(f"{x:}, {y:}")


def reverse_text(text: Text) -> Text:
    return text[::-1]


@pytest.mark.parametrize("model", [
    count,
    create_text,
    reverse_text
])
def test_push_example_model(model, acumos_session, acumos_options, model_name_prefix, request):
    name = request.node.name

    acumos_session.push(Model(f=model), model_name_prefix + name.split("[")[1][:-1], options=acumos_options)

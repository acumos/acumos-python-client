import pytest
import os
import sys
from acumos.session import AcumosSession, Options


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

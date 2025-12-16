# tests/test_packages_exists.py

import importlib
import pytest

MODULES = ["mlir"]


@pytest.mark.parametrize("name", MODULES)
def test_can_import(name):
    try:
        importlib.import_module(name)
    except Exception as e:
        pytest.fail(f"import {name} failed：{e!r}. Please refer to README.md")

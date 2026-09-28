import ast
import inspect
import json
from pathlib import Path

import pytest

import MultiSuSiE

REPO_ROOT = Path(__file__).resolve().parents[1]
API_USAGE_PATHS = [
    REPO_ROOT / "tests.py",
    REPO_ROOT / "examples" / "example_for_install_script.py",
    REPO_ROOT / "examples" / "example.ipynb",
]


def _code_sources(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        return [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
    return [path.read_text()]


@pytest.mark.parametrize("path", API_USAGE_PATHS, ids=lambda path: path.name)
def test_multisusie_rss_calls_use_supported_keywords(path):
    supported_keywords = set(inspect.signature(MultiSuSiE.multisusie_rss).parameters)
    used_keywords = set()
    call_count = 0

    for source in _code_sources(path):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "multisusie_rss"
            ):
                call_count += 1
                used_keywords.update(
                    keyword.arg for keyword in node.keywords if keyword.arg is not None
                )

    assert call_count > 0
    assert used_keywords <= supported_keywords

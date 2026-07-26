from __future__ import annotations

from pathlib import Path

import pytest

from georeliab_mve import toml_compat

try:
    import tomllib as stdlib_tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 verification path
    stdlib_tomllib = None


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "pyproject.toml",
    ROOT / "configs" / "a100_real_mve_overlay.toml",
    ROOT / "configs" / "dual_mve_protocol.toml",
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_python_310_fallback_matches_stdlib_for_committed_toml(path: Path) -> None:
    if stdlib_tomllib is None:
        pytest.skip("stdlib tomllib is unavailable before Python 3.11")
    text = path.read_text(encoding="utf-8")
    assert toml_compat._loads_fallback(text) == stdlib_tomllib.loads(text)


def test_fallback_supports_nested_tables_comments_and_multiline_arrays() -> None:
    text = r'''
title = "literal # text"
[execution.tuning]
enabled = true
devices = [
  "cuda:0",
  "cuda:1", # shared rendering devices
]
[[resource]]
name = 'first'
weight = 2.5
[[resource]]
name = 'second'
weight = 4
'''
    expected = {
        "title": "literal # text",
        "execution": {
            "tuning": {"enabled": True, "devices": ["cuda:0", "cuda:1"]}
        },
        "resource": [
            {"name": "first", "weight": 2.5},
            {"name": "second", "weight": 4},
        ],
    }
    assert toml_compat._loads_fallback(text) == expected
    if stdlib_tomllib is not None:
        assert expected == stdlib_tomllib.loads(text)


@pytest.mark.parametrize(
    "text",
    (
        "value = 1\nvalue = 2\n",
        "[runtime]\n[runtime]\n",
        "value = { unsupported = true }\n",
        "value = 2026-07-26\n",
    ),
)
def test_fallback_rejects_ambiguous_or_unsupported_constructs(text: str) -> None:
    with pytest.raises(ValueError):
        toml_compat._loads_fallback(text)


def test_a100_shells_do_not_depend_on_external_tomli() -> None:
    scripts = (ROOT / "scripts" / "a100").glob("*.sh")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
    assert "import tomli" not in combined
    assert "toml_compat" in combined

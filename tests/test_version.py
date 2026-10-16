"""``sqgraph.__version__`` has a single source: the version of the installed distribution (pyproject.toml)."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest

import sqgraph
from sqgraph.memory import cli


def test_version_is_the_installed_distribution_version():
    try:
        dist_version = importlib.metadata.version("sqgraph")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sqgraph is not installed (running from a bare checkout)")
    assert sqgraph.__version__ == dist_version


def test_version_falls_back_when_not_installed(monkeypatch):
    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        importlib.reload(sqgraph)
        assert sqgraph.__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(sqgraph)


def test_cli_version_flag_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exc:  # argparse's version action prints and exits
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"sqgraph-memory {sqgraph.__version__}"

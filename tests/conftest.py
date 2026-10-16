from __future__ import annotations

import pytest

from sqgraph import GraphDB


@pytest.fixture(autouse=True)
def _isolate_user_dirs(tmp_path_factory, monkeypatch):
    """Point every per-user location at a throw-away directory, so no test can touch the real agent configs
    (~/.claude, ~/.codex, %APPDATA%\\Claude, ~/.config/opencode, ...) or the real ~/.sqgraph-memory."""
    fake = tmp_path_factory.mktemp("user")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(fake))
    monkeypatch.setenv("APPDATA", str(fake / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(fake / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake / ".config"))
    monkeypatch.setenv("SQGRAPH_MEMORY_HOME", str(fake / ".sqgraph-memory"))
    for var in ("SQGRAPH_MEMORY_DB", "SQGRAPH_MEMORY_GROUP", "SQGRAPH_MEMORY_EXTRACTOR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def db():
    d = GraphDB(":memory:")
    try:
        yield d
    finally:
        d.close()


@pytest.fixture
def social(db):
    """Small social graph used by many Cypher tests.

    Alice(30) -KNOWS{since:2015}-> Bob(25) -KNOWS{since:2018}-> Carol(35) -KNOWS-> Alice (cycle)
    Alice -WORKS_AT-> Acme ; Bob -WORKS_AT-> Acme ; Carol -WORKS_AT-> Globex
    Dave(40) isolated (no relationships)
    """
    ids = {}
    ids["alice"] = db.create_node(["Person", "Employee"], {"name": "Alice", "age": 30, "active": True}).id
    ids["bob"] = db.create_node("Person", {"name": "Bob", "age": 25, "active": False}).id
    ids["carol"] = db.create_node("Person", {"name": "Carol", "age": 35, "tags": ["a", "b"]}).id
    ids["dave"] = db.create_node("Person", {"name": "Dave", "age": 40}).id
    ids["acme"] = db.create_node("Company", {"name": "Acme"}).id
    ids["globex"] = db.create_node("Company", {"name": "Globex"}).id
    db.create_edge(ids["alice"], ids["bob"], "KNOWS", {"since": 2015})
    db.create_edge(ids["bob"], ids["carol"], "KNOWS", {"since": 2018})
    db.create_edge(ids["carol"], ids["alice"], "KNOWS", {"since": 2020})
    db.create_edge(ids["alice"], ids["acme"], "WORKS_AT")
    db.create_edge(ids["bob"], ids["acme"], "WORKS_AT")
    db.create_edge(ids["carol"], ids["globex"], "WORKS_AT")
    return ids

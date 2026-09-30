from __future__ import annotations

import pytest

from sqgraph import GraphDB


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

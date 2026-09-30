"""Differential tests: Cypher results (compiled to SQL) against brute-force Python on random graphs."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import pytest

from sqgraph import GraphDB


def build(seed: int, n_nodes: int = 9, n_edges: int = 16):
    rnd = random.Random(seed)
    db = GraphDB()
    nodes = {}
    for i in range(n_nodes):
        label = rnd.choice(["A", "B"])
        nodes[i] = {"label": label, "v": rnd.randint(0, 6)}
    ids = {i: db.create_node(nodes[i]["label"], {"i": i, "v": nodes[i]["v"]}).id for i in nodes}
    rev = {v: k for k, v in ids.items()}
    edges = []  # (eid, src, dst, type, w) in terms of index i
    for _ in range(n_edges):
        s, d = rnd.randrange(n_nodes), rnd.randrange(n_nodes)  # self loops / parallel edges allowed
        t = rnd.choice(["R", "S"])
        w = rnd.randint(0, 3)
        e = db.create_edge(ids[s], ids[d], t, {"w": w})
        edges.append((e.id, s, d, t, w))
    return db, nodes, edges, rev


def oriented(edge, undirected):
    """Yield (from, to) orientations of an edge as a pattern step would traverse it."""
    _, s, d, _, _ = edge
    yield s, d
    if undirected and s != d:
        yield d, s


@pytest.mark.parametrize("seed", range(8))
def test_two_hop_patterns_with_uniqueness(seed):
    db, nodes, edges, _ = build(seed)
    for undirected in (False, True):
        arrow = "-[r1:R]-" if undirected else "-[r1:R]->"
        arrow2 = "-[r2:R]-" if undirected else "-[r2:R]->"
        got = db.query(f"MATCH (a){arrow}(b){arrow2}(c) RETURN a.i AS a, b.i AS b, c.i AS c")
        expect = []
        for e1 in edges:
            if e1[3] != "R":
                continue
            for a, b in oriented(e1, undirected):
                for e2 in edges:
                    if e2[3] != "R" or e2[0] == e1[0]:
                        continue
                    for b2, c in oriented(e2, undirected):
                        if b2 == b:
                            expect.append((a, b, c))
        assert Counter((r["a"], r["b"], r["c"]) for r in got) == Counter(expect), (seed, undirected)
    db.close()


@pytest.mark.parametrize("seed", range(8))
def test_filtered_join_with_where_pushdown(seed):
    db, nodes, edges, _ = build(seed)
    got = db.query(
        "MATCH (a:A)-[r:R]->(b:B)-[s:S]->(c) WHERE a.v > 1 AND c.v < 5 AND b.v <> a.v AND r.w >= s.w RETURN a.i AS a, b.i AS b, c.i AS c"
    )
    expect = []
    for e1 in edges:
        _, a, b, t, w1 = e1
        if t != "R" or nodes[a]["label"] != "A" or nodes[b]["label"] != "B":
            continue
        for e2 in edges:
            _, b2, c, t2, w2 = e2
            if t2 != "S" or b2 != b:
                continue
            if nodes[a]["v"] > 1 and nodes[c]["v"] < 5 and nodes[b]["v"] != nodes[a]["v"] and w1 >= w2:
                expect.append((a, b, c))
    assert Counter((r["a"], r["b"], r["c"]) for r in got) == Counter(expect)
    db.close()


@pytest.mark.parametrize("seed", range(8))
def test_optional_match_and_aggregation(seed):
    db, nodes, edges, _ = build(seed)
    got = db.query(
        "MATCH (a:A) OPTIONAL MATCH (a)-[r:R]->(b) RETURN a.i AS a, count(b) AS n, count(*) AS rows, sum(b.v) AS total, collect(b.i) AS bs"
    )
    by = defaultdict(list)
    for _, s, d, t, _ in edges:
        if t == "R":
            by[s].append(d)
    for row in got:
        a = row["a"]
        out = by[a]
        assert row["n"] == len(out)
        assert row["rows"] == max(1, len(out))
        assert row["total"] == sum(nodes[b]["v"] for b in out)
        assert sorted(row["bs"]) == sorted(out)
    assert {r["a"] for r in got} == {i for i, n in nodes.items() if n["label"] == "A"}
    db.close()


@pytest.mark.parametrize("seed", range(8))
def test_grouping_ordering_and_limits(seed):
    db, nodes, edges, _ = build(seed)
    got = db.query(
        "MATCH (a)-[r]->(b) RETURN b.v AS v, type(r) AS t, count(*) AS c, max(r.w) AS mw ORDER BY c DESC, v ASC, t ASC LIMIT 5"
    )
    agg: dict = defaultdict(lambda: [0, -1])
    for _, s, d, t, w in edges:
        key = (nodes[d]["v"], t)
        agg[key][0] += 1
        agg[key][1] = max(agg[key][1], w)
    expect = sorted(((c, v, t, mw) for (v, t), (c, mw) in agg.items()), key=lambda x: (-x[0], x[1], x[2]))[:5]
    assert [(r["c"], r["v"], r["t"], r["mw"]) for r in got] == expect
    db.close()


@pytest.mark.parametrize("seed", range(6))
def test_bound_variable_reuse_and_cycles(seed):
    db, nodes, edges, _ = build(seed, n_nodes=7, n_edges=14)
    got = db.query("MATCH (a)-[:R]->(b)-[:R]->(a) RETURN count(*) AS c")[0]["c"]
    rs = [(s, d) for _, s, d, t, _ in edges if t == "R"]
    expect = 0
    for i, (s1, d1) in enumerate(rs):
        for j, (s2, d2) in enumerate(rs):
            if i != j and s2 == d1 and d2 == s1:
                expect += 1
    assert got == expect
    got = db.query("MATCH (a:A) WITH a MATCH (a)-[r:S]-(b) RETURN count(*) AS c")[0]["c"]
    expect = 0
    for _, s, d, t, _ in edges:
        if t != "S":
            continue
        if nodes[s]["label"] == "A":
            expect += 1
        if s != d and nodes[d]["label"] == "A":
            expect += 1
    assert got == expect
    db.close()


@pytest.mark.parametrize("seed", range(6))
def test_exists_subquery_and_not_exists(seed):
    db, nodes, edges, _ = build(seed)
    got = {r["i"] for r in db.query("MATCH (a) WHERE EXISTS { MATCH (a)-[:R]->(:B) } RETURN a.i AS i")}
    expect = {s for _, s, d, t, _ in edges if t == "R" and nodes[d]["label"] == "B"}
    assert got == expect
    got = {r["i"] for r in db.query("MATCH (a) WHERE NOT EXISTS { (a)-[]-() } RETURN a.i AS i")}
    touched = {x for _, s, d, _, _ in edges for x in (s, d)}
    assert got == set(nodes) - touched
    db.close()


def test_write_then_read_consistency_random():
    rnd = random.Random(7)
    with GraphDB() as db:
        model: dict[int, int] = {}
        for _ in range(200):
            k = rnd.randrange(15)
            op = rnd.choice(["merge", "inc", "del"])
            if op == "merge":
                db.query("MERGE (n:K {k: $k}) ON CREATE SET n.c = 1 ON MATCH SET n.c = n.c + 1", {"k": k})
                model[k] = model.get(k, 0) + 1
            elif op == "inc":
                db.query("MATCH (n:K {k: $k}) SET n.c = n.c + 10", {"k": k})
                if k in model:
                    model[k] += 10
            else:
                db.query("MATCH (n:K {k: $k}) DELETE n", {"k": k})
                model.pop(k, None)
        got = {r["k"]: r["c"] for r in db.query("MATCH (n:K) RETURN n.k AS k, n.c AS c")}
        assert got == model


@pytest.mark.parametrize("seed", range(6))
def test_fixed_hop_followed_by_varlen_respects_uniqueness(seed):
    db, nodes, edges, _ = build(seed, n_nodes=7, n_edges=13)
    r_edges = [e for e in edges if e[3] == "R"]

    def trails(start, used, lo, hi):
        found = []

        def rec(node, used_, depth):
            if lo <= depth <= hi:
                found.append(node)
            if depth == hi:
                return
            for e in r_edges:
                if e[0] in used_:
                    continue
                if e[1] == node:
                    rec(e[2], used_ | {e[0]}, depth + 1)

        rec(start, frozenset(used), 0)
        return found

    got = db.query("MATCH (a)-[r:R]->(b)-[:R*1..2]->(c) RETURN a.i AS a, b.i AS b, c.i AS c")
    expect = []
    for e in r_edges:
        for c in trails(e[2], {e[0]}, 1, 2):
            expect.append((e[1], e[2], c))
    assert Counter((r["a"], r["b"], r["c"]) for r in got) == Counter(expect)

    got = db.query("MATCH (a)-[:R*1..2]->(b)-[s:R]->(c) RETURN a.i AS a, b.i AS b, c.i AS c")
    expect = []

    # brute force: enumerate trails of length 1..2 with their edge sets, then append a distinct fixed edge
    def trails_with_edges(start, lo, hi):
        out = []

        def rec(node, used_, depth):
            if lo <= depth <= hi:
                out.append((node, used_))
            if depth == hi:
                return
            for e in r_edges:
                if e[0] in used_:
                    continue
                if e[1] == node:
                    rec(e[2], used_ | {e[0]}, depth + 1)

        rec(start, frozenset(), 0)
        return out

    for a in nodes:
        for b, used in trails_with_edges(a, 1, 2):
            for s in r_edges:
                if s[0] not in used and s[1] == b:
                    expect.append((a, b, s[2]))
    assert Counter((r["a"], r["b"], r["c"]) for r in got) == Counter(expect)
    db.close()

"""Cross-process write lock of sqgraph.memory (FileLock and its use by Memory)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from sqgraph.memory import FakeEmbedder, FakeLLM, Memory
from sqgraph.memory.filelock import FileLock

G = "guild"

_HOLDER = """
import sys
from sqgraph.memory.filelock import FileLock

lock = FileLock(sys.argv[1], timeout=5)
lock.acquire()
print("locked", flush=True)
sys.stdin.read()  # hold the lock until the parent closes stdin (or dies)
"""


def hold_in_other_process(path):
    """A child process holding ``FileLock(path)``; returns the Popen once the lock is taken."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "locked"
    return proc


def release_other_process(proc):
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0


def test_file_lock_excludes_other_holders_in_the_same_process(tmp_path):
    path = tmp_path / "x.lock"
    a, b = FileLock(path), FileLock(path, timeout=0.2)
    with a:
        assert a.locked
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="x.lock"):
            b.acquire()
        assert 0.15 <= time.monotonic() - t0 < 2
        with pytest.raises(TimeoutError):
            b.acquire(timeout=0)
        with pytest.raises(RuntimeError):
            a.acquire()  # not re-entrant
    assert not a.locked
    with b:  # free again
        assert b.locked
    b.release()  # releasing twice is harmless
    assert path.exists()  # the lock file is left in place


def test_file_lock_waits_until_released(tmp_path):
    path = tmp_path / "x.lock"
    a = FileLock(path)
    a.acquire()
    threading.Timer(0.3, a.release).start()
    t0 = time.monotonic()
    with FileLock(path, timeout=5):
        assert 0.2 <= time.monotonic() - t0 < 4


def test_file_lock_across_processes(tmp_path):
    path = tmp_path / "x.lock"
    proc = hold_in_other_process(path)
    try:
        with pytest.raises(TimeoutError, match="another process"):
            FileLock(path, timeout=0.2).acquire()
    finally:
        release_other_process(proc)  # the process ends: the OS drops its lock
    with FileLock(path, timeout=2):
        pass


def test_lock_is_released_when_the_holder_dies(tmp_path):
    path = tmp_path / "x.lock"
    proc = hold_in_other_process(path)
    proc.kill()
    proc.wait(timeout=10)
    with FileLock(path, timeout=2):
        pass


def test_memory_write_waits_for_another_process_and_times_out(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G, lock_timeout=0.3) as mem:
        proc = hold_in_other_process(str(path) + ".lock")
        try:
            t0 = time.monotonic()
            with pytest.raises(TimeoutError, match="mem.db.lock"):
                mem.add_episode("[[Rails|Service]] {{alice|USES|Rails}}", author="alice")
            assert time.monotonic() - t0 < 3
            with pytest.raises(TimeoutError):
                mem.purge()
            with pytest.raises(TimeoutError):
                mem.forget_episode("ep_x")
            assert mem.search("Rails") == []  # reads never need the lock
            assert mem.stats(G)["nodes"]["Episode"] == 0  # and nothing was written
        finally:
            release_other_process(proc)
        r = mem.add_episode("[[Rails|Service]] {{alice|USES|Rails}}", author="alice")
        assert len(r.facts_created) == 1
        assert mem._lock_depth == 0 and not mem._file_lock.locked  # released after every operation


def test_memory_open_waits_for_the_lock_too(tmp_path):
    path = tmp_path / "mem.db"
    proc = hold_in_other_process(str(path) + ".lock")
    try:
        with pytest.raises(TimeoutError):
            Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G, lock_timeout=0.2)  # schema setup is a write
    finally:
        release_other_process(proc)
    Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G, lock_timeout=0.2).close()


def test_two_memories_on_one_file_serialize_whole_ingestions(tmp_path):
    path = tmp_path / "mem.db"
    events: list[tuple[str, str]] = []

    def slow(name, llm):
        base = llm._task_extract_entities

        def extract(payload):
            events.append((name, "start"))
            time.sleep(0.25)  # an "LLM call" inside the ingestion
            out = base(payload)
            events.append((name, "end"))
            return out

        return extract

    llm_a, llm_b = FakeLLM(), FakeLLM()
    llm_a.script["extract_entities"] = slow("a", llm_a)
    llm_b.script["extract_entities"] = slow("b", llm_b)
    a = Memory(path, llm_a, FakeEmbedder(), default_group_id=G)
    b = Memory(path, llm_b, FakeEmbedder(), default_group_id=G)
    errors: list[BaseException] = []

    def work(mem, who):
        try:
            # both introduce the same brand-new entity
            mem.add_episode(f"[[Rails|Service]] {{{{{who}|USES|Rails}}}} [[{who}|Person]]", author=who, source="t", channel="c")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(a, "a")), threading.Thread(target=work, args=(b, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    # no interleaving: each ingestion ran start..end back to back
    assert events[0][0] == events[1][0] and events[2][0] == events[3][0] and events[0][0] != events[2][0]
    assert [e[1] for e in events] == ["start", "end", "start", "end"]
    # the second one found the entity the first one created instead of creating its own
    rows = a.db.query("MATCH (e:Entity {name:'Rails'}) RETURN e")
    assert len(rows) == 1
    assert a.stats(G)["nodes"]["Episode"] == 2
    a.close()
    b.close()


def test_guard_is_reentrant_and_releases_the_file_lock_once(tmp_path):
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G, lock_timeout=0.3) as mem:
        with mem._write_guard():
            assert mem._file_lock.locked and mem._lock_depth == 1
            mem.add_episode("[[Rails|Service]] {{alice|USES|Rails}}", author="alice")  # nested, no deadlock
            assert mem._file_lock.locked and mem._lock_depth == 1
            with pytest.raises(TimeoutError):
                FileLock(str(path) + ".lock", timeout=0.1).acquire()
        assert not mem._file_lock.locked and mem._lock_depth == 0
        with FileLock(str(path) + ".lock", timeout=1):
            pass
        # an exception inside the guard still releases everything
        with pytest.raises(RuntimeError):
            with mem._write_guard():
                raise RuntimeError("boom")
        assert not mem._file_lock.locked and mem._lock_depth == 0


def test_no_file_lock_for_in_memory_and_read_only_databases(tmp_path):
    with Memory(":memory:", FakeLLM(), FakeEmbedder(), default_group_id=G) as mem:
        assert mem._file_lock is None
        mem.add_episode("[[Rails|Service]] {{alice|USES|Rails}}", author="alice")
    path = tmp_path / "mem.db"
    with Memory(path, FakeLLM(), FakeEmbedder(), default_group_id=G):
        pass
    with Memory(path, None, None, default_group_id=G, read_only=True) as ro:
        assert ro._file_lock is None
    assert (tmp_path / "mem.db.lock").exists()

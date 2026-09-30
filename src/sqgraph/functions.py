"""Python user-defined SQL functions registered on every connection.

They back the parts of Cypher that plain SQLite cannot express (regular
expressions, Unicode case mapping, dynamically typed ``+``, list concatenation,
...).  Every function is deterministic and NULL-propagating unless noted.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from functools import lru_cache
from typing import Any


last_error: list[str] = []  # message of the most recent error raised by a UDF (sqlite3 hides it)


class UdfError(Exception):
    """Raised by a UDF to abort the statement with a readable message."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        last_error[:] = [message]


def _divzero() -> None:
    raise UdfError("/ by zero (integer division or modulo by zero)")


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.DOTALL)
    except re.error as exc:
        raise UdfError(f"Invalid regular expression {pattern!r}: {exc}") from exc


def _regexp(pattern: Any, value: Any) -> int | None:
    """SQL ``value REGEXP pattern`` (search semantics, as in most SQL dialects)."""
    if pattern is None or value is None:
        return None
    return 1 if _compile(str(pattern)).search(str(value)) else 0


def _matches(value: Any, pattern: Any) -> int | None:
    """Cypher ``value =~ pattern``: the regular expression must match the whole string."""
    if pattern is None or value is None:
        return None
    if not isinstance(value, str) or not isinstance(pattern, str):
        return None
    return 1 if _compile(pattern).fullmatch(value) else 0


def _lower(v: Any) -> Any:
    return v.lower() if isinstance(v, str) else v


def _upper(v: Any) -> Any:
    return v.upper() if isinstance(v, str) else v


def _to_text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return f"{v:.1f}"
        return repr(v)
    return str(v)


def _add(a: Any, b: Any) -> Any:
    """Dynamically typed ``+`` on native SQLite values (numbers or strings)."""
    if a is None or b is None:
        return None
    if isinstance(a, str) or isinstance(b, str):
        return _to_text(a) + _to_text(b)  # type: ignore[operator]
    return a + b


def _mod(a: Any, b: Any) -> Any:
    if a is None or b is None or b == 0:
        return None
    if isinstance(a, int) and isinstance(b, int):
        return int(math.fmod(a, b))
    return math.fmod(a, b)


def _pow(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    try:
        return float(a) ** float(b)
    except (OverflowError, ZeroDivisionError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            try:
                f = float(v.strip())
                return None if (math.isnan(f) or math.isinf(f)) else int(f)
            except ValueError:
                return None
    return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _jconcat(a: Any, b: Any) -> str | None:
    """Concatenate two JSON arrays (given as JSON text)."""
    if a is None or b is None:
        return None
    return json.dumps(json.loads(a) + json.loads(b), separators=(",", ":"), ensure_ascii=False)


def _jappend(a: Any, b: Any) -> str | None:
    """Append a JSON-encoded element ``b`` to the JSON array ``a``."""
    if a is None or b is None:
        return None
    return json.dumps(json.loads(a) + [json.loads(b)], separators=(",", ":"), ensure_ascii=False)


def _jprepend(a: Any, b: Any) -> str | None:
    """Prepend a JSON-encoded element ``a`` to the JSON array ``b``."""
    if a is None or b is None:
        return None
    return json.dumps([json.loads(a)] + json.loads(b), separators=(",", ":"), ensure_ascii=False)


def _jslice(v: Any, lo: Any, hi: Any) -> str | None:
    """``list[lo..hi]`` with Cypher/Python slice semantics on a JSON array."""
    if v is None:
        return None
    return json.dumps(json.loads(v)[lo:hi], separators=(",", ":"), ensure_ascii=False)


def _mapproj(base: Any, overlay: Any) -> str | None:
    """Overlay ``overlay`` (JSON object text) onto ``base`` (JSON object text); nulls are kept."""
    if base is None or overlay is None:
        return None
    d = json.loads(base)
    d.update(json.loads(overlay))
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def _split(s: Any, delim: Any) -> str | None:
    if s is None or delim is None:
        return None
    parts = str(s).split(str(delim)) if delim != "" else list(str(s))
    return json.dumps(parts, ensure_ascii=False)


def _reverse(v: Any) -> Any:
    if isinstance(v, str):
        return v[::-1]
    return v


def _jreverse(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(list(reversed(json.loads(v))), separators=(",", ":"), ensure_ascii=False)


def _jtail(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(json.loads(v)[1:], separators=(",", ":"), ensure_ascii=False)


def _math1(fn: Any) -> Any:
    def wrapper(v: Any) -> Any:
        if v is None:
            return None
        try:
            return fn(float(v))
        except (ValueError, OverflowError):
            return None

    return wrapper


def _sign(v: Any) -> Any:
    if v is None:
        return None
    return (v > 0) - (v < 0)


def _round(v: Any, digits: Any = 0) -> Any:
    """Round half away from zero (Neo4j's ``round``); always returns a float."""
    if v is None:
        return None
    d = int(digits or 0)
    m = 10.0**d
    x = float(v) * m
    r = math.floor(abs(x) + 0.5)
    return math.copysign(r, x) / m


def _tostr(v: Any) -> Any:
    return _to_text(v)


def _rand() -> float:
    import random

    return random.random()


def register(conn: sqlite3.Connection) -> None:
    """Register all functions on ``conn``."""
    f = lambda name, n, fn, det=True: conn.create_function(name, n, fn, deterministic=det)  # noqa: E731
    f("regexp", 2, _regexp)
    f("sqg_matches", 2, _matches)
    f("sqg_lower", 1, _lower)
    f("sqg_upper", 1, _upper)
    f("sqg_add", 2, _add)
    f("sqg_mod", 2, _mod)
    f("sqg_pow", 2, _pow)
    f("sqg_toint", 1, _to_int)
    f("sqg_tofloat", 1, _to_float)
    f("sqg_tostr", 1, _tostr)
    f("sqg_jconcat", 2, _jconcat)
    f("sqg_jappend", 2, _jappend)
    f("sqg_jprepend", 2, _jprepend)
    f("sqg_split", 2, _split)
    f("sqg_reverse", 1, _reverse)
    f("sqg_jreverse", 1, _jreverse)
    f("sqg_jtail", 1, _jtail)
    f("sqg_jslice", 3, _jslice)
    f("sqg_mapproj", 2, _mapproj)
    f("sqg_ceil", 1, _math1(lambda x: float(math.ceil(x))))
    f("sqg_floor", 1, _math1(lambda x: float(math.floor(x))))
    f("sqg_sqrt", 1, _math1(math.sqrt))
    f("sqg_exp", 1, _math1(math.exp))
    f("sqg_log", 1, _math1(math.log))
    f("sqg_log10", 1, _math1(math.log10))
    f("sqg_sin", 1, _math1(math.sin))
    f("sqg_cos", 1, _math1(math.cos))
    f("sqg_sign", 1, _sign)
    f("sqg_round", 1, _round)
    f("sqg_round", 2, _round)
    f("sqg_rand", 0, _rand, False)
    f("sqg_divzero", 0, _divzero, False)

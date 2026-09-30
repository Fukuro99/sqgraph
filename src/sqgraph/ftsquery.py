"""Translate Lucene-style query strings (as used by Neo4j fulltext indexes) to FTS5 syntax.

Supported: bare terms (combined with OR by default, like Lucene), ``"phrases"``,
``AND`` / ``OR`` / ``NOT`` (also ``&&`` ``||`` ``!``), parentheses, ``-term``
exclusion, ``term*`` prefix search and ``field:term`` column filters.
Everything else is treated as plain text, so user input can never produce an
FTS5 syntax error through special characters alone.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r'"([^"]*)"|(\()|(\))|([^\s()"]+)')
_OPERATORS = {"AND": "AND", "OR": "OR", "NOT": "NOT", "&&": "AND", "||": "OR", "!": "NOT"}


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def _clean_word(word: str) -> str:
    # strip Lucene fuzzy/boost suffixes and characters FTS5 would treat specially
    word = re.sub(r"[~^]\d*(\.\d+)?$", "", word)
    return word


def lucene_to_fts5(query: str, columns: dict[str, str] | None = None) -> str:
    """Convert ``query`` to an FTS5 MATCH expression (empty string if nothing searchable).

    ``columns`` maps user-visible property names to FTS5 column names for ``field:term``.
    """
    columns = columns or {}
    out: list[str] = []
    excluded: list[str] = []
    prev_operand = False
    pending_exclude = False
    for m in _TOKEN_RE.finditer(query):
        phrase, lp, rp, word = m.groups()
        piece: str | None = None
        if lp:
            if prev_operand:
                out.append("OR")
            out.append("(")
            prev_operand = False
            continue
        if rp:
            out.append(")")
            prev_operand = True
            continue
        if phrase is not None:
            if not phrase.strip():
                continue
            piece = _quote(phrase)
        else:
            assert word is not None
            if word in _OPERATORS:
                op = _OPERATORS[word]
                if op == "NOT" and not prev_operand:
                    pending_exclude = True
                    continue
                out.append(op)
                prev_operand = False
                continue
            negate = False
            if word[0] in "-!" and len(word) > 1:
                negate = True
                word = word[1:]
            elif word[0] == "+" and len(word) > 1:
                word = word[1:]
            colname = None
            if ":" in word:
                field, _, rest = word.partition(":")
                if field in columns and rest:
                    colname = columns[field]
                    word = rest
            prefix = word.endswith("*")
            word = _clean_word(word.rstrip("*"))
            if not word:
                continue
            piece = _quote(word) + ("*" if prefix else "")
            if colname:
                piece = f"{colname} : {piece}"
            if negate:
                excluded.append(piece)
                continue
        if pending_exclude:
            excluded.append(piece)
            pending_exclude = False
            continue
        if prev_operand:
            out.append("OR")
        out.append(piece)
        prev_operand = True
    # drop dangling operators / empty groups
    text = " ".join(out)
    text = re.sub(r"\(\s*\)", "", text).strip()
    while True:
        new = re.sub(r"^(AND|OR|NOT)\s+|\s+(AND|OR|NOT)$", "", text).strip()
        if new == text:
            break
        text = new
    if not text:
        return ""
    if excluded:
        text = f"({text}) NOT ({' OR '.join(excluded)})"
    return text


def fallback_fts5(query: str) -> str:
    """Last-resort translation: every word quoted, joined with OR."""
    words = [w for w in re.findall(r"[^\s()\"]+", query) if w.upper() not in _OPERATORS]
    return " OR ".join(_quote(w) for w in words)

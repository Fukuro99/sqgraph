"""Text / time helpers shared by the memory modules (no LLM, no DB)."""

from __future__ import annotations

import re
import secrets
import unicodedata
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Iterable

JST = timezone(timedelta(hours=9), "JST")
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- ids


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


# --------------------------------------------------------------------------- names


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


_HONORIFIC_RE = re.compile(r"(さん|くん|君|ちゃん|様|氏|san)$")


def norm_key(name: str) -> str:
    """Comparison key of an entity name / alias: NFKC, case-folded, without separators or honorifics."""
    s = nfkc(name).casefold().strip()
    s = re.sub(r"[\s_\-・/／.]+", "", s)
    s2 = _HONORIFIC_RE.sub("", s)
    return s2 or s


# Display names in chat tools carry temporary status suffixes:
#   "alice(9/14-9/16お休み)", "name [不在]9/9～11", "name_1718休"
_ABSENCE_RES = [
    re.compile(r"\s*[\(（][^)）]*(?:お休み|休み|不在|休暇|OOO|ooo)[^)）]*[\)）]"),
    re.compile(r"\s*[\[【][^\]】]*(?:不在|お休み|休み|休暇|OOO|ooo)[^\]】]*[\]】]\S*"),
    re.compile(r"[_\-\s]?\d{3,4}休"),
]


def clean_display_name(name: str) -> str:
    """Strip vacation / absence markers so one person keeps one stable name."""
    name = name or ""
    prev = None
    while prev != name:
        prev = name
        for rx in _ABSENCE_RES:
            name = rx.sub("", name).strip()
    return name or "unknown"


def person_forms(raw_name: str) -> tuple[str, list[str]]:
    """Return ``(canonical_name, aliases)`` for a chat display name.

    ``"taro/Yamada_1718休"`` -> ``("taro/Yamada", ["taro", "Yamada", "taro/Yamada_1718休"])``.
    The raw form and each ``/``- or ``|``-separated part (at least 2 characters) become aliases.
    """
    raw = (raw_name or "").strip()
    canonical = clean_display_name(raw)
    aliases: list[str] = []
    if raw and raw != canonical:
        aliases.append(raw)
    parts = [p.strip() for p in re.split(r"\s*[/／|｜]\s*", canonical) if len(p.strip()) >= 2]
    if len(parts) > 1:
        aliases.extend(parts)
    return canonical, list(dict.fromkeys(a for a in aliases if a != canonical))


def clean_message(text: str) -> str:
    """Remove absence markers glued to @mentions and normalise line breaks."""
    text = re.sub(r"(@[^\s(（]+)[\(（][^)）]*(?:お休み|不在|休み)[^)）]*[\)）]", r"\1", text)
    return text.strip()


# --------------------------------------------------------------------------- query terms

_TERM_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*|[゠-ヿㇰ-ㇿ]+|[一-鿿々〆ヵヶ]+")


def query_terms(query: str) -> list[str]:
    """Split a (Japanese) question into content-word candidates.

    Hiragana runs (mostly particles / inflections) are dropped; ASCII words, katakana runs and kanji
    runs are kept (NFKC + case-folded). Used for trigram fulltext and entity-name matching.
    """
    q = nfkc(query).casefold()
    terms = [t.strip("._-") for t in _TERM_RE.findall(q)]
    return list(dict.fromkeys(t for t in terms if len(t) >= 2))


def fts_or_query(terms: Iterable[str], min_len: int = 3) -> str:
    """OR of quoted phrases for the (trigram) fulltext indexes; terms shorter than ``min_len`` are skipped."""
    ts = [t.replace('"', " ").strip() for t in terms if len(t.strip()) >= min_len]
    return " OR ".join(f'"{t}"' for t in ts)


# --------------------------------------------------------------------------- time


def resolve_tz(tz: str | tzinfo | None) -> tzinfo:
    if tz is None:
        return JST
    if isinstance(tz, tzinfo):
        return tz
    if tz.upper() in ("JST", "ASIA/TOKYO"):
        return JST
    if tz.upper() == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo

    return ZoneInfo(tz)


def parse_time(value: Any, tz: tzinfo = JST) -> datetime | None:
    """Parse an ISO-ish timestamp (``date``, ``datetime``, ``YYYY-MM-DD[ T]HH:MM[:SS][Z|+09:00]``).

    Naive values are interpreted in ``tz``; date-only values mean local midnight. Returns an aware
    datetime or ``None`` for empty / unparseable input.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        s = nfkc(str(value)).strip().replace("/", "-")
        s = re.sub(r"\s+", " ", s)
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        s = re.sub(r"(?<=\d)\s+(?=\d{2}:)", "T", s, count=1)
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def to_iso(dt: datetime) -> str:
    """Canonical stored form: UTC, seconds precision, ``Z`` suffix (lexicographically sortable)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_or_none(value: Any, tz: tzinfo = JST) -> str | None:
    dt = parse_time(value, tz)
    return to_iso(dt) if dt else None


def now_iso() -> str:
    return to_iso(datetime.now(timezone.utc))


def to_local(iso: str | None, tz: tzinfo = JST) -> str | None:
    """Stored UTC ISO string -> local ISO string with offset (``2026-07-01T18:52:00+09:00``)."""
    if not iso:
        return None
    dt = parse_time(iso, timezone.utc)
    return dt.astimezone(tz).isoformat(timespec="seconds") if dt else iso


def local_with_weekday(iso: str, tz: tzinfo = JST) -> str:
    dt = parse_time(iso, timezone.utc)
    assert dt is not None
    dt = dt.astimezone(tz)
    return f"{dt.strftime('%Y-%m-%d %H:%M')} ({_WEEKDAYS[dt.weekday()]}, {dt.strftime('%z')[:3]}:{dt.strftime('%z')[3:]})"


def short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

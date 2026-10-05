"""Pluggable LLM / embedding providers for :mod:`sqgraph.memory`.

The core ``sqgraph`` database is LLM-free; everything model related lives here behind two tiny
protocols so that the memory layer can be used with OpenAI (default), another vendor, or a
deterministic fake (tests).

* :class:`LLMProvider` -- ``complete_json`` (schema-constrained JSON) and ``complete_text``.
* :class:`Embedder` -- ``embed(texts) -> vectors`` with a fixed ``dimensions``.
* :class:`OpenAILLM` / :class:`OpenAIEmbedder` -- OpenAI implementations with token/cost tracking
  through a shared :class:`UsageTracker`. ``OpenAIEmbedder(base_url=...)`` talks to any OpenAI-compatible
  ``/v1/embeddings`` server (Ollama, LM Studio, llama.cpp, TEI, vLLM, ...), e.g. one on the LAN;
  :func:`probe_dimensions` finds out how many dimensions such a model produces.
* :class:`FakeLLM` / :class:`FakeEmbedder` -- offline, deterministic implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_DIMS = 1536
PLACEHOLDER_API_KEY = "sqgraph-local"  # the openai client insists on a non-empty key; most local servers ignore it
CUSTOM_EMBED_TIMEOUT = 30.0  # seconds, for custom (LAN) endpoints
CUSTOM_EMBED_CONNECT_TIMEOUT = 5.0  # an unreachable LAN host should fail fast
CUSTOM_EMBED_RETRIES = 1
CUSTOM_EMBED_BATCH = 32  # e.g. text-embeddings-inference rejects bigger client batches by default

# USD per 1M tokens: (input, cached input, output)
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.13, 0.0),
}
JPY_PER_USD = 150  # rough, for the human-readable summary only


# --------------------------------------------------------------------------- protocols


@runtime_checkable
class LLMProvider(Protocol):
    """A chat model able to return JSON that conforms to a JSON schema."""

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, task: str = "") -> dict[str, Any]:
        """Return the parsed JSON object. ``user`` is a JSON document (the task input)."""
        ...

    def complete_text(self, system: str, user: str, *, task: str = "") -> str:
        """Return free text."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """A text embedding model."""

    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector (``dimensions`` floats) per input text, in order."""
        ...


# --------------------------------------------------------------------------- usage tracking


class UsageTracker:
    """Thread-safe token / cost accumulator keyed by ``(phase, model, task)``.

    Set :attr:`phase` (e.g. ``"ingest"`` / ``"query"``) to group costs; :meth:`snapshot` /
    :meth:`diff_usd` measure the cost of a code block.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "-"
        self.rows: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "inp": 0, "cached": 0, "out": 0}
        )

    def add(self, model: str, inp: int, cached: int, out: int, task: str = "") -> None:
        with self._lock:
            r = self.rows[(self.phase, model, task)]
            r["calls"] += 1
            r["inp"] += int(inp)
            r["cached"] += int(cached)
            r["out"] += int(out)

    @staticmethod
    def usd(model: str, r: dict[str, int]) -> float | None:
        if model not in PRICES:
            return None
        pin, pcached, pout = PRICES[model]
        return ((r["inp"] - r["cached"]) * pin + r["cached"] * pcached + r["out"] * pout) / 1e6

    def total_usd(self, phase: str | None = None) -> float:
        with self._lock:
            return sum(
                self.usd(m, r) or 0.0 for (ph, m, _), r in self.rows.items() if phase is None or ph == phase
            )

    def total_calls(self, phase: str | None = None) -> int:
        with self._lock:
            return sum(r["calls"] for (ph, _, _), r in self.rows.items() if phase is None or ph == phase)

    def snapshot(self) -> float:
        """Current total USD (use with :meth:`diff_usd`)."""
        return self.total_usd()

    def diff_usd(self, snapshot: float) -> float:
        return self.total_usd() - snapshot

    def by_task(self, phase: str | None = None) -> dict[str, dict[str, float]]:
        """``{task: {calls, inp, out, usd}}`` (optionally for one phase)."""
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            for (ph, model, task), r in self.rows.items():
                if phase is not None and ph != phase:
                    continue
                d = out.setdefault(task or model, {"calls": 0, "inp": 0, "out": 0, "usd": 0.0})
                d["calls"] += r["calls"]
                d["inp"] += r["inp"]
                d["out"] += r["out"]
                d["usd"] += self.usd(model, r) or 0.0
        return out

    def summary(self, title: str = "") -> str:
        lines = [f"=== OpenAI cost{': ' + title if title else ''} ==="]
        lines.append(f"  {'phase':<14}{'model':<24}{'calls':>6}{'in tok':>9}{'(cached)':>10}{'out tok':>9}{'USD':>10}")
        merged: dict[tuple[str, str], dict[str, int]] = {}
        with self._lock:
            for (ph, model, _), r in self.rows.items():
                m = merged.setdefault((ph, model), {"calls": 0, "inp": 0, "cached": 0, "out": 0})
                for k in m:
                    m[k] += r[k]
        for (ph, model), r in sorted(merged.items()):
            u = self.usd(model, r)
            cost = f"{u:>10.5f}" if u is not None else f"{'?':>10}"
            lines.append(f"  {ph:<14}{model:<24}{r['calls']:>6}{r['inp']:>9}{r['cached']:>10}{r['out']:>9}{cost}")
        tot = self.total_usd()
        lines.append(
            f"  TOTAL: {self.total_calls()} calls, ${tot:.5f} (approx. {tot * JPY_PER_USD:.2f} JPY at {JPY_PER_USD} JPY/USD)"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- OpenAI


def load_env(path: str | os.PathLike[str] | None = None) -> None:
    """Load ``KEY=VALUE`` lines of a ``.env`` file into ``os.environ`` (existing variables win).

    Without ``path`` the current directory and its parents are searched for ``.env``. Values are
    never printed or logged.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        here = Path.cwd().resolve()
        candidates += [d / ".env" for d in (here, *here.parents)]
    for env in candidates:
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
                if m and not line.lstrip().startswith("#"):
                    os.environ.setdefault(m.group(1), m.group(2).strip().strip("'\""))
            return


def _openai_client(client: Any = None, **kwargs: Any) -> Any:
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('OpenAI providers need the "openai" package: pip install "sqgraph[memory]"') from exc
    return OpenAI(max_retries=kwargs.pop("max_retries", 4), **kwargs)


class OpenAILLM:
    """Chat completions with strict JSON-schema structured output and cost tracking."""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        tracker: UsageTracker | None = None,
        client: Any = None,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.tracker = tracker or UsageTracker()
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self._client = _openai_client(client, **client_kwargs)

    def _create(self, system: str, user: str, task: str, **kwargs: Any) -> Any:
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        u = getattr(r, "usage", None)
        if u:
            details = getattr(u, "prompt_tokens_details", None)
            self.tracker.add(
                self.model, u.prompt_tokens, getattr(details, "cached_tokens", 0) or 0, u.completion_tokens, task
            )
        return r

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, task: str = "") -> dict[str, Any]:
        r = self._create(
            system,
            user,
            task,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": task or "result", "strict": True, "schema": schema},
            },
        )
        return json.loads(r.choices[0].message.content or "{}")

    def complete_text(self, system: str, user: str, *, task: str = "") -> str:
        r = self._create(system, user, task)
        return (r.choices[0].message.content or "").strip()


class EmbedderError(RuntimeError):
    """An embedding server returned something unusable (wrong count / dimension of vectors)."""

    def __init__(self, message: str, actual: int | None = None) -> None:
        super().__init__(message)
        self.actual = actual  # the dimension the server really produced (when that was the problem)


def normalize_base_url(url: str) -> str:
    """``http://host:11434`` -> ``http://host:11434/v1`` (a bare origin gets ``/v1``); trailing slashes dropped."""
    from urllib.parse import urlsplit

    u = url.strip().rstrip("/")
    if not u:
        raise ValueError("empty base URL")
    parts = urlsplit(u)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"invalid base URL {url!r}: expected http(s)://host[:port][/v1]")
    return u + "/v1" if not parts.path else u


class OpenAIEmbedder:
    """OpenAI embeddings (default ``text-embedding-3-small``, 1536 dimensions) or any OpenAI-compatible server.

    ``base_url`` points the client at another ``/v1/embeddings`` endpoint (Ollama ``http://host:11434/v1``,
    LM Studio ``http://host:1234/v1``, llama.cpp, TEI, vLLM, ...). For such endpoints:

    * ``dimensions`` is *not* sent unless ``send_dimensions=True`` (many servers reject the parameter), so
      ``dimensions`` must be what the model produces (see :func:`probe_dimensions`), and ``encoding_format="float"``
      is requested explicitly;
    * ``OPENAI_API_KEY`` is never used: ``api_key`` (else a placeholder) is sent instead;
    * the timeout defaults to 30 s (5 s to connect), retries to 1 and ``batch_size`` to 32.

    Returned vectors are checked: the count must match the inputs and each must have ``dimensions`` floats
    (:class:`EmbedderError` otherwise). ``dimensions=None`` skips that check and learns the dimension from the first
    response (used by :func:`probe_dimensions`).
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        dimensions: int | None = DEFAULT_EMBED_DIMS,
        tracker: UsageTracker | None = None,
        client: Any = None,
        batch_size: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        send_dimensions: bool | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.dimensions = dimensions  # type: ignore[assignment]
        self.tracker = tracker or UsageTracker()
        self.base_url = normalize_base_url(base_url) if base_url else None
        self.send_dimensions = send_dimensions
        self.batch_size = batch_size or (CUSTOM_EMBED_BATCH if self.base_url else 128)
        if self.base_url:
            from openai import Timeout  # needs the openai package anyway (the client below)

            t = CUSTOM_EMBED_TIMEOUT if timeout is None else float(timeout)
            client_kwargs.update(
                base_url=self.base_url,
                api_key=api_key or PLACEHOLDER_API_KEY,  # never fall back to OPENAI_API_KEY: it must not reach a LAN host
                timeout=Timeout(t, connect=min(CUSTOM_EMBED_CONNECT_TIMEOUT, t)),
                max_retries=CUSTOM_EMBED_RETRIES if max_retries is None else max_retries,
            )
        else:
            if api_key:
                client_kwargs["api_key"] = api_key
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
        self._client = _openai_client(client, **client_kwargs)

    @property
    def endpoint(self) -> str:
        """Identity of the server: the base URL, or ``"openai"`` for the OpenAI API."""
        return self.base_url or "openai"

    def _sends_dimensions(self) -> bool:
        if self.send_dimensions is not None:
            return self.send_dimensions
        if self.base_url:
            return False
        return self.model != DEFAULT_EMBED_MODEL or self.dimensions != DEFAULT_EMBED_DIMS

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        texts = [t if t.strip() else "." for t in texts]
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i : i + self.batch_size])
            kwargs: dict[str, Any] = {}
            if self._sends_dimensions() and self.dimensions:
                kwargs["dimensions"] = self.dimensions
            if self.base_url:
                kwargs["encoding_format"] = "float"  # the SDK would ask for base64, which not every server supports
            r = self._client.embeddings.create(model=self.model, input=chunk, **kwargs)
            usage = getattr(r, "usage", None)
            if usage:
                self.tracker.add(self.model, getattr(usage, "total_tokens", 0) or 0, 0, 0, "embed")
            data = list(r.data)
            if len(data) != len(chunk):
                raise EmbedderError(
                    f"embedding server {self.endpoint} returned {len(data)} vectors for {len(chunk)} texts "
                    f"(model {self.model!r})"
                )
            if all(isinstance(getattr(d, "index", None), int) for d in data):
                data.sort(key=lambda d: d.index)
            vecs = [list(d.embedding) for d in data]
            if self.dimensions is None and vecs:
                self.dimensions = len(vecs[0])  # type: ignore[assignment]
            for v in vecs:
                if len(v) != self.dimensions:
                    raise EmbedderError(
                        f"embedding server {self.endpoint} returned {len(v)}-dimensional vectors for model "
                        f"{self.model!r}, but {self.dimensions} dimensions are configured: set the dimensions to "
                        f"{len(v)} (embed_dimensions) or use another model",
                        actual=len(v),
                    )
            out += vecs
        return out


def probe_dimensions(embedder_or_params: Any = None, **params: Any) -> int:
    """Embed one short text and return the length of the vector (the model's dimensions).

    ``embedder_or_params`` is an embedder (anything with ``embed``) or a mapping of :class:`OpenAIEmbedder`
    arguments (``{"model": ..., "base_url": ..., "api_key": ...}``; keyword arguments work too). A dimension
    configured on an :class:`OpenAIEmbedder` is not enforced here: the *actual* dimension is returned.
    """
    if embedder_or_params is None or isinstance(embedder_or_params, Mapping):
        args = {**(embedder_or_params or {}), **params}
        args["dimensions"] = None
        embedder: Any = OpenAIEmbedder(**args)
    else:
        embedder = embedder_or_params
    try:
        vecs = embedder.embed(["sqgraph-memory dimension probe"])
    except EmbedderError as exc:
        if exc.actual is not None:
            return exc.actual
        raise
    if not vecs or not len(vecs[0]):
        raise EmbedderError("the embedder returned an empty vector")
    return len(vecs[0])


def build_embedder(
    model: str = DEFAULT_EMBED_MODEL,
    *,
    base_url: str | None = None,
    dimensions: int | None = None,
    existing_dims: int | None = None,
    api_key: str | None = None,
    tracker: UsageTracker | None = None,
) -> tuple[OpenAIEmbedder, bool]:
    """An :class:`OpenAIEmbedder` for the configured endpoint, and whether its dimensions had to be probed.

    Dimensions: ``dimensions`` (configured), else the ``existing_dims`` of the database, else -- for the OpenAI API --
    1536, or for a custom ``base_url`` a probe of the server (one embedding call; :class:`EmbedderError` /
    connection errors propagate).
    """
    probed = False
    dims = dimensions or existing_dims
    if not dims:
        if base_url:
            dims = probe_dimensions(model=model, base_url=base_url, api_key=api_key)
            probed = True
        else:
            dims = DEFAULT_EMBED_DIMS
    return OpenAIEmbedder(model, dimensions=dims, tracker=tracker, base_url=base_url, api_key=api_key), probed


# --------------------------------------------------------------------------- fakes (tests / offline)


class FakeEmbedder:
    """Deterministic embedder: hashed character trigrams, L2-normalised.

    Texts sharing substrings get similar vectors, which is enough to exercise vector search
    without a network.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            vec = [0.0] * self.dimensions
            s = re.sub(r"\s+", "", t.casefold())
            grams = [s[i : i + 3] for i in range(max(1, len(s) - 2))] or [s or "."]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
                vec[h % self.dimensions] += 1.0 if (h >> 32) & 1 else -1.0
                vec[(h >> 8) % self.dimensions] += 0.5
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


_FAKE_ENTITY_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_FAKE_FACT_RE = re.compile(r"\{\{([^}]*)\}\}")


class FakeLLM:
    """Deterministic, rule-based stand-in for an LLM, driven by a tiny markup in the message text.

    * ``[[Name|Type]]`` marks an entity (``Type`` defaults to ``Concept``).
    * ``{{Subject|RELATION|Object|valid_at|invalid_at}}`` marks a fact (``Object``, ``valid_at`` and
      ``invalid_at`` may be empty; the fact sentence is ``"Subject RELATION Object"``).

    Behaviour of the other steps: entity resolution matches candidates by normalised name/alias;
    fact de-duplication treats the same (subject, relation, object) as a duplicate and the same
    (subject, relation) with a different object as a contradiction. ``script`` (``{task: fn(payload)
    -> dict}``) overrides any step. All calls are recorded in :attr:`calls`.
    """

    def __init__(self, script: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None) -> None:
        self.script = dict(script or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # -- protocol
    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, task: str = "") -> dict[str, Any]:
        payload = json.loads(user)
        self.calls.append((task, payload))
        fn = self.script.get(task) or getattr(self, f"_task_{task}", None)
        if fn is None:
            raise NotImplementedError(f"FakeLLM has no handler for task {task!r}")
        return fn(payload)

    def complete_text(self, system: str, user: str, *, task: str = "") -> str:
        self.calls.append((task, {"user": user}))
        fn = self.script.get(task)
        if fn is not None:
            return fn({"user": user})  # type: ignore[return-value]
        return "FAKE ANSWER\n" + user[:2000]

    def count(self, task: str) -> int:
        return sum(1 for t, _ in self.calls if t == task)

    # -- tasks
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[\s_\-・/]+", "", s.casefold())

    def _task_extract_entities(self, p: dict[str, Any]) -> dict[str, Any]:
        text = p["message"]
        ents, seen = [], set()
        for m in _FAKE_ENTITY_RE.finditer(text):
            name, typ = m.group(1).strip(), (m.group(2) or "Concept").strip()
            if name not in seen:
                seen.add(name)
                ents.append({"name": name, "type": typ, "description": f"{name} ({typ})"})
        return {"entities": ents, "has_facts": bool(_FAKE_FACT_RE.search(text))}

    def _task_resolve_entities(self, p: dict[str, Any]) -> dict[str, Any]:
        out = []
        for e in p["entities"]:
            key = self._norm(e["name"])
            match = None
            for c in e["candidates"]:
                if key in {self._norm(c["name"]), *(self._norm(a) for a in c.get("aliases", []))}:
                    match = c["id"]
                    break
            out.append({"idx": e["idx"], "match": match})
        return {"resolutions": out}

    def _task_extract_facts(self, p: dict[str, Any]) -> dict[str, Any]:
        by_name = {self._norm(e["name"]): e["idx"] for e in p["entities"]}
        for e in p["entities"]:
            for a in e.get("aliases", []):
                by_name.setdefault(self._norm(a), e["idx"])
        facts = []
        for m in _FAKE_FACT_RE.finditer(p["message"]):
            parts = [x.strip() for x in m.group(1).split("|")] + [""] * 5
            subj, rel, obj, va, ia = parts[:5]
            si = by_name.get(self._norm(subj))
            if si is None:
                continue
            oi = by_name.get(self._norm(obj)) if obj else None
            sn = next(e["name"] for e in p["entities"] if e["idx"] == si)
            on = next((e["name"] for e in p["entities"] if e["idx"] == oi), "") if oi is not None else ""
            facts.append(
                {
                    "subject": si,
                    "relation": rel or "RELATES_TO",
                    "object": oi,
                    "fact": f"{sn} {rel} {on}".strip(),
                    "valid_at": va or None,
                    "invalid_at": ia or None,
                    "modality": "definite",
                }
            )
        used = sorted({f["subject"] for f in facts} | {f["object"] for f in facts if f["object"] is not None})
        summaries = []
        for i in used:
            e = next(x for x in p["entities"] if x["idx"] == i)
            new = "; ".join(f["fact"] for f in facts if i in (f["subject"], f["object"]))
            old = (e.get("summary") or "").strip()
            summaries.append({"idx": i, "summary": f"{old} | {new}" if old else new})
        return {"facts": facts, "entity_summaries": summaries}

    def _task_dedupe_facts(self, p: dict[str, Any]) -> dict[str, Any]:
        out = []
        for nf in p["new_facts"]:
            dup, contra = None, []
            for ex in p["existing_facts"]:
                same_subj = ex["subject"] == nf["subject"]
                if same_subj and ex["relation"] == nf["relation"] and ex["object"] == nf["object"]:
                    dup = ex["h"]
                    break
                if same_subj and ex["relation"] == nf["relation"] and ex["object"] != nf["object"]:
                    contra.append(ex["h"])
            out.append({"idx": nf["idx"], "duplicate_of": dup, "contradicts": [] if dup else contra})
        return {"results": out}

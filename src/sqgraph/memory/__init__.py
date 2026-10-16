"""Temporal knowledge-graph long-term memory for agents, built on sqgraph.

>>> from sqgraph.memory import Memory, OpenAILLM, OpenAIEmbedder
>>> mem = Memory("memory.db", OpenAILLM(), OpenAIEmbedder(), default_group_id="my-guild")
>>> mem.add_episode("Bob owns the auth service", channel="backend", author="alice")
>>> mem.search("Who owns the auth service?", channels=["backend"])

The core :mod:`sqgraph` package stays LLM-free; models are plugged in through the
:class:`LLMProvider` / :class:`Embedder` protocols (OpenAI implementations and deterministic fakes
are included). See ``docs/memory.md`` for the data model, ACL design and MCP server.
"""

from .core import MEMORY_SCHEMA_VERSION, Answer, EpisodeRef, EpisodeResult, FactHit, ForgetResult, Memory
from .providers import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_DIMS,
    DEFAULT_EMBED_MODEL,
    Embedder,
    EmbedderError,
    FakeEmbedder,
    FakeLLM,
    LLMProvider,
    OpenAIEmbedder,
    OpenAILLM,
    UsageTracker,
    build_embedder,
    load_env,
    probe_dimensions,
)
from .util import clean_display_name, person_forms

__all__ = [
    "Memory",
    "FactHit",
    "EpisodeRef",
    "EpisodeResult",
    "ForgetResult",
    "MEMORY_SCHEMA_VERSION",
    "Answer",
    "LLMProvider",
    "Embedder",
    "EmbedderError",
    "OpenAILLM",
    "OpenAIEmbedder",
    "FakeLLM",
    "FakeEmbedder",
    "UsageTracker",
    "load_env",
    "probe_dimensions",
    "build_embedder",
    "clean_display_name",
    "person_forms",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_EMBED_DIMS",
]

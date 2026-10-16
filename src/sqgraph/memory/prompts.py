"""Prompts and JSON schemas of the ingestion / answering pipeline.

Every user message sent to the model is a JSON document (the task input); the system prompt holds the
instructions. Keep these short: they are paid for on every episode.
"""

from __future__ import annotations

from typing import Any

ENTITY_TYPES = ["Person", "Service", "Repository", "Team", "Organization", "Ticket", "Project", "Event", "Place", "Concept", "Other"]

# What is worth remembering; shared by every extraction prompt so the server-side and agent-side rules agree.
DURABLE_KNOWLEDGE = (
    "who owns or works on what, status, decisions, plans and schedules, dependencies, where things are kept, absences, "
    "problems and their causes, and the user's preferences, standing instructions to the assistant, personal facts and habits"
)

# Shared naming / reference rules for entities.
_ENTITY_NAMING = """- name: the canonical, most complete name, without honorifics or titles. A product version is the product unless the version itself is the topic ("Rails", not "Rails 7.1"). Name tickets and PRs with their repository or project ("api-server PR #42"), never a bare number.
- Resolve pronouns and vague references ("it", "that issue") to the concrete entity using the earlier messages and the author; never output the pronoun itself."""

# Shared rules for the temporal fields of a fact.
_FACT_TIME = """- valid_at: when the fact starts being true, as ISO 8601 local time (a date alone is fine), computed from the message time ("from next week", "tomorrow", "on the 14th"; take the nearest sensible year). Only if the message states or implies a start; otherwise null (= from the message time). Never guess.
- invalid_at: when it stops being true, if the message says so: the first moment it no longer holds (away 14-16 March -> 17 March), else null.
- modality: definite | planned (future intention or schedule) | tentative (unsure, proposal)."""


def obj_schema(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _nullable(t: str) -> dict[str, Any]:
    return {"type": [t, "null"]}


# --------------------------------------------------------------------------- 1. extract entities

EXTRACT_ENTITIES_SYSTEM = f"""You build a knowledge graph from chat messages and notes (team chat, or a user's notes to their AI assistant), in any language.
Task: list the entities the CURRENT message talks about, and say whether it contains facts worth remembering.

Entity types: {", ".join(ENTITY_TYPES)}.
- Extract people, services/systems/repositories/tools, teams/organizations, tickets/PRs, projects, events/meetings, places and important concepts/policies.
{_ENTITY_NAMING} The earlier messages are in `context`.
- Do NOT extract the author (added automatically), greetings, generic nouns without a specific name ("the meeting", "a task"), dates, or entities that appear only in `context`.
- description: one short phrase saying what the entity is, as far as this message tells.
- has_facts: true if the message states or implies durable knowledge: {DURABLE_KNOWLEDGE}. false for chit-chat, thanks, acknowledgements and questions that add no information."""

EXTRACT_ENTITIES_SCHEMA = obj_schema(
    entities={
        "type": "array",
        "items": obj_schema(
            name={"type": "string"},
            type={"type": "string", "enum": ENTITY_TYPES},
            description={"type": "string"},
        ),
    },
    has_facts={"type": "boolean"},
)

# --------------------------------------------------------------------------- 2. resolve entities

RESOLVE_ENTITIES_SYSTEM = """You deduplicate knowledge-graph entities.
For each new entity, decide which existing candidate (if any) is the SAME real-world thing. Consider aliases, abbreviations, transliterations and other spellings or scripts, and the descriptions.
Return match = the candidate `id`, or null if it is a different / new entity. Different versions, different repos or different people with similar names are NOT the same. When unsure, return null."""

RESOLVE_ENTITIES_SCHEMA = obj_schema(
    resolutions={
        "type": "array",
        "items": obj_schema(idx={"type": "integer"}, match=_nullable("string")),
    }
)

# --------------------------------------------------------------------------- 3. extract facts

EXTRACT_FACTS_SYSTEM = f"""You extract temporal facts for a knowledge graph from ONE chat message or note, in any language.
Input: `entities` (idx, name, type, aliases, current summary), `author`, `occurred_at` (the message time: local time with weekday and UTC offset), `context` (earlier messages, only for resolving references), `known_relations`, `message`.

Output `facts`: relations between two entities (subject idx -> object idx) or attributes of one entity (object null).
- Record only durable knowledge stated or clearly implied by `message`: {DURABLE_KNOWLEDGE}. Not facts that appear only in `context` or in summaries, not that someone asked, wondered or said thanks, not chit-chat. Usually 0-5 facts; none is fine.
- fact: one self-contained sentence in the message's language that names its entities (no pronouns), e.g. "Bob owns access control in the auth service".
- relation: UPPER_SNAKE_CASE verb phrase (WORKS_ON, OWNS, MANAGES, DEPENDS_ON, HAS_STATUS, PLANS_TO, ABSENT, DECIDED, USES, PREFERS ...). Reuse a name from `known_relations` (the graph's vocabulary, most used first) whenever one fits; use the active direction (OWNS, not OWNED_BY).
{_FACT_TIME}
- A change of state (new owner, new status) is an ordinary new fact; contradictions with existing facts are resolved later.
`entity_summaries`: for entities that got new durable information, a one-line summary (at most 120 characters, in the message's language) merging the old summary with the new information; omit the others. Keep transient details such as absence dates out of summaries."""

EXTRACT_FACTS_SCHEMA = obj_schema(
    facts={
        "type": "array",
        "items": obj_schema(
            subject={"type": "integer"},
            relation={"type": "string"},
            object=_nullable("integer"),
            fact={"type": "string"},
            valid_at=_nullable("string"),
            invalid_at=_nullable("string"),
            modality={"type": "string", "enum": ["definite", "planned", "tentative"]},
        ),
    },
    entity_summaries={
        "type": "array",
        "items": obj_schema(idx={"type": "integer"}, summary={"type": "string"}),
    },
)

# --------------------------------------------------------------------------- 4. dedupe / contradiction

DEDUPE_FACTS_SYSTEM = """You maintain a temporal knowledge graph. For each NEW fact decide against the EXISTING valid facts (handles h1, h2, ...):
- duplicate_of: the handle of an existing fact that says the same thing (same meaning, same state), else null.
- contradicts: handles of existing facts that can no longer be true once the new fact holds (e.g. a new owner replaces the old owner, a status changes, a plan is cancelled or replaced). Facts that merely add other information, or are compatible/parallel (a person can work on several things), are NOT contradicted.
A duplicate never contradicts anything. Be conservative: only list clear conflicts."""

DEDUPE_FACTS_SCHEMA = obj_schema(
    results={
        "type": "array",
        "items": obj_schema(
            idx={"type": "integer"},
            duplicate_of=_nullable("string"),
            contradicts={"type": "array", "items": {"type": "string"}},
        ),
    }
)

# --------------------------------------------------------------------------- agent extraction (no server-side LLM)

# Given to the calling agent by ``Memory.prepare_episode``: the agent does steps 1-4 above itself.
AGENT_EXTRACTION_INSTRUCTIONS = f"""You are the extractor. Turn `context.message` into the `extraction` JSON (schema in `schema`) and send it with `token` to commit. Use only what the message states or clearly implies; never invent facts.

ENTITIES (`entities`): the people, services/tools/repos, teams/orgs, tickets/PRs, projects, events, places and key concepts/policies the message is about. Types: {", ".join(ENTITY_TYPES)}.
- ref: your handle ("e1", "e2", ...). The author (`context.author_ref`) already exists: reference it as "author", never list it.
{_ENTITY_NAMING} The earlier messages are in `context.recent`. Skip greetings and generic nouns.
- existing_id: if an entry of `context.candidate_entities` is the SAME real-world thing (alias, abbreviation, other spelling), set its id and do not create a duplicate; else null. Different versions, repos or people with similar names are different.
- summary: optional one line (at most 120 characters, in the message's language) merging the old summary with the new durable information. aliases: other spellings seen in the message.

FACTS (`facts`): durable knowledge only: {DURABLE_KNOWLEDGE}. Not chit-chat, thanks, questions without new information, or this session's transient details. Usually 0-5 facts; none is fine.
- subject/object: entity refs. An attribute of one entity (a status, an absence, a preference) has object null.
- fact: ONE self-contained sentence in the message's language that names its entities (no pronouns), e.g. "Bob owns access control in the auth service".
- relation: UPPER_SNAKE_CASE (OWNS, WORKS_ON, MANAGES, DEPENDS_ON, HAS_STATUS, PLANS_TO, ABSENT, DECIDED, USES, PREFERS ...). Reuse a name from `context.known_relations` (the memory's vocabulary, most used first) whenever one fits the meaning; coin a new one only when none does. Use the active direction (Bob OWNS repo, not repo OWNED_BY Bob).
- The message time is `context.message.occurred_at` (with weekday and UTC offset).
{_FACT_TIME}
- confidence: 0-1 (default by modality).
- duplicate_of: id of a fact in `context.related_facts` that this one merely restates (same meaning, same state), else null. A duplicate contradicts nothing.
- contradicts: ids of facts in `context.related_facts` that can no longer be true once this one holds (new owner, changed status, cancelled plan). Compatible or parallel facts are NOT contradicted; list only clear conflicts. The old fact is closed, never deleted.
Finally call commit_memory(token, extraction) (CLI: `sqgraph-memory commit`); if it reports problems, fix the JSON and call it again with the same token."""

AGENT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": 'your handle for this entity, e.g. "e1" ("author" is reserved)'},
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "existing_id": {"type": ["string", "null"], "description": "id of the matching entry of candidate_entities, else null"},
                    "summary": {"type": ["string", "null"]},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["ref", "name", "type"],
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": 'entity ref, or "author"'},
                    "object": {"type": ["string", "null"], "description": "entity ref, or null for an attribute"},
                    "relation": {"type": "string", "description": "UPPER_SNAKE_CASE"},
                    "fact": {"type": "string", "description": "one self-contained sentence"},
                    "valid_at": {"type": ["string", "null"]},
                    "invalid_at": {"type": ["string", "null"]},
                    "modality": {"type": "string", "enum": ["definite", "planned", "tentative"]},
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "duplicate_of": {"type": ["string", "null"], "description": "id of a related_facts entry this restates"},
                    "contradicts": {"type": "array", "items": {"type": "string"}, "description": "ids of related_facts entries that are no longer true"},
                },
                "required": ["subject", "relation", "fact"],
            },
        },
    },
}

# --------------------------------------------------------------------------- answer

DEFAULT_ANSWER_LANGUAGE = "English"


def answer_system(language: str | None = DEFAULT_ANSWER_LANGUAGE) -> str:
    """System prompt of ``Memory.answer``; ``language=None`` answers in the language of the question."""
    lang = language.strip() if language and language.strip() else "the language of the question"
    return f"""You answer questions from a long-term memory (a temporal knowledge graph).
Answer concisely in {lang}, using only the given `facts` (and `entities`).
- Cite the sources of each claim as [F1]; add the author and date when it helps.
- Each fact has a status: valid (true now), ended (no longer true; a past state) or scheduled (planned, not started yet). For the current state prefer valid facts; for history, use ended facts too and explain in chronological order.
- Use valid_at / invalid_at to say from when until when something holds. If `as_of` is set, answer for that point in time.
- If the facts do not answer the question, say so instead of guessing. Point out contradictions and uncertainty (modality tentative or planned)."""


ANSWER_SYSTEM = answer_system()

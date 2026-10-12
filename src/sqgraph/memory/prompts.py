"""Prompts and JSON schemas of the ingestion / answering pipeline.

Every user message sent to the model is a JSON document (the task input); the system prompt holds the
instructions. Keep these short: they are paid for on every episode.
"""

from __future__ import annotations

from typing import Any

ENTITY_TYPES = ["Person", "Service", "Repository", "Team", "Organization", "Ticket", "Project", "Event", "Place", "Concept", "Other"]


def obj_schema(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _nullable(t: str) -> dict[str, Any]:
    return {"type": [t, "null"]}


# --------------------------------------------------------------------------- 1. extract entities

EXTRACT_ENTITIES_SYSTEM = f"""You build a knowledge graph from chat messages or notes (team chat, or a user's notes to their AI assistant; mostly Japanese).
Task: list the entities the CURRENT message talks about, and say whether it contains memorable facts.

Entity types: {", ".join(ENTITY_TYPES)}.
Rules:
- Extract people (use the name as written, without honorifics like さん), services/systems/repos/tools/software versions worth tracking, teams/orgs, tickets/PR numbers, projects, events/meetings, places (cities, venues, offices), and important concepts/policies.
- Resolve pronouns and vague references (それ, あれ, this, 例の件) to the concrete entity using `context` (earlier messages) and `author`. Never output the pronoun itself.
- Use the most complete canonical name (e.g. "notify-bot", "Ruby on Rails"; a version of a product is the product: output "Rails", not "Rails 7.1", unless the version itself is the topic). Name PRs/issues with their repo or project ("iac-terraform PR #278"), never a bare number.
- Do NOT extract: the author (added automatically), greetings, generic nouns (タスク, 会議 without a specific name), dates, or entities only present in `context`.
- description: one short phrase about what the entity is, as far as this message tells.
- has_facts: true if the message states or implies something worth remembering later (who owns/does what, status, decision, plan, schedule, absence, dependency, problem, request assignment; a user's preferences, standing instructions to the assistant like 〜してほしい / 〜は使わない, personal facts such as where they live or what they plan, habits). false for pure chit-chat, thanks, acknowledgements, questions without new info."""

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
For each new entity, decide which existing candidate (if any) is the SAME real-world thing. Consider aliases, abbreviations, romanised/Japanese spellings, and the descriptions.
Return match = the candidate `id`, or null if it is a different / new entity. Different versions, different repos or different people with similar names are NOT the same. When unsure, return null."""

RESOLVE_ENTITIES_SCHEMA = obj_schema(
    resolutions={
        "type": "array",
        "items": obj_schema(idx={"type": "integer"}, match=_nullable("string")),
    }
)

# --------------------------------------------------------------------------- 3. extract facts

EXTRACT_FACTS_SYSTEM = """You extract temporal facts for a knowledge graph from ONE chat message or note (team chat or a user's notes to their AI assistant; mostly Japanese).
Input: `entities` (idx, name, type, current summary), `author`, `occurred_at` (local time, weekday, UTC offset), `context` (earlier messages, only for resolving references), `message`.

Output `facts`: relations between entities (subject idx -> object idx) or attributes of one entity (object null).
- Only facts stated or clearly implied by `message`. Do not repeat facts that only appear in `context` or in summaries.
- `fact`: one self-contained sentence in the message language that names its entities (no pronouns), e.g. "Bob は認証基盤の権限管理を担当している".
- `relation`: UPPER_SNAKE_CASE verb phrase (WORKS_ON, OWNS, MANAGES, DEPENDS_ON, HAS_STATUS, PLANS_TO, ABSENT, DECIDED, USES, HANDS_OVER_TO ...). Reuse a name from `known_relations` (the graph's vocabulary, most used first) whenever one fits; use the active direction (OWNS, not OWNED_BY).
- valid_at: when the fact starts being true in the world, ISO 8601 (local time, no timezone, date-only is fine). Give a value ONLY if the message states or implies a start time (dates, 来週から, 明日, 9/14, 先月...), computed from `occurred_at` (assume the nearest sensible year). Otherwise null (= true from `occurred_at` on). Never guess a date.
- invalid_at: when the fact stops being true if the message says so (e.g. absence 9/14-9/16 -> 2026-09-17: the first moment it is no longer true), else null.
- modality: definite | planned (future intention/schedule) | tentative (unsure, proposal, question-like).
- A statement that changes an earlier state (new owner, new status) is a normal new fact; contradictions are handled later.
- Record durable knowledge: ownership/assignment, responsibility, status, decisions, plans/schedules, dependencies, where things are managed/stored, absences, problems and their causes, and a user's preferences, standing instructions to the assistant (e.g. "ユーザーは回答を簡潔にしてほしい"), personal facts (residence, plans) and habits. Do NOT record that someone asked / wonders / thanks / suggests looking something up, and skip chit-chat. Prefer 0-5 high-value facts.
`entity_summaries`: for entities that got NEW durable information, an updated summary (max 120 chars, in Japanese) merging the old summary with the new info; omit others. Do not put transient details (absence dates) in summaries."""

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
- name: the canonical, most complete name without honorifics (Rails, not "Rails 7.1"; "iac-terraform PR #278", not "#278"). Resolve pronouns and vague words (それ, 例の件) using `context.recent`. Skip greetings and generic nouns.
- existing_id: if an entry of `context.candidate_entities` is the SAME real-world thing (alias, abbreviation, other spelling), set its id and do not create a duplicate; else null. Different versions, repos or people with similar names are different.
- summary: optional one line (<=120 chars, message language) merging the old summary with new durable info. aliases: other spellings seen in the message.

FACTS (`facts`): durable knowledge only: ownership/assignment, status, decisions, plans/schedules, dependencies, where things live, absences, problems and causes, and the user's preferences, standing instructions and personal facts/habits. Not chit-chat, thanks, questions without new info, or this session's transient details. Prefer 0-5 high-value facts; none is fine.
- subject/object: entity refs. An attribute of one entity (a status, an absence, a preference) has object null.
- fact: ONE self-contained sentence in the message language that names its entities (no pronouns), e.g. "Bob は認証基盤の権限管理を担当している".
- relation: UPPER_SNAKE_CASE (OWNS, WORKS_ON, MANAGES, DEPENDS_ON, HAS_STATUS, PLANS_TO, ABSENT, DECIDED, USES, PREFERS ...). Reuse a name from `context.known_relations` (the memory's vocabulary, most used first) whenever one fits the meaning; coin a new one only when none does. Use the active direction (Bob OWNS repo, not repo OWNED_BY Bob).
- valid_at: when it starts being true, computed from `context.message.occurred_at` (weekday and UTC offset are given): ISO 8601, local time, date-only is fine (来週から, 明日, 9/14 ...). null means the message time. Never guess a date.
- invalid_at: when it stops being true if stated (absence 9/14-9/16 -> 2026-09-17, the first moment it is no longer true), else null.
- modality: definite | planned (future intention/schedule) | tentative (unsure, proposal). confidence: 0-1 (default by modality).
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

ANSWER_SYSTEM = """あなたは社内の長期記憶(時系列ナレッジグラフ)を使って質問に答えるアシスタントです。
与えられた `facts`(と `entities`)だけを根拠に、日本語で簡潔に答えてください。
- 各主張の末尾に出典を [F1] のように付ける。必要なら送信者・日時も添える。
- 各 fact には status があります: valid=現在有効 / ended=すでに失効(過去の状態) / scheduled=まだ始まっていない予定。今の状態を聞かれたら valid を優先し、経緯を聞かれたら ended も使って時系列で説明する。
- valid_at / invalid_at (有効期間) を踏まえ、いつからいつまでの話かを明示する。`as_of` が指定されている場合はその時点の状況として答える。
- 情報が無い・不十分な場合は推測せず、分からないと述べる。矛盾や不確かさ (modality=tentative/planned) があれば明記する。"""

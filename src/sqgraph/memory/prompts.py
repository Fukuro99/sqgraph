"""Prompts and JSON schemas of the ingestion / answering pipeline.

Every user message sent to the model is a JSON document (the task input); the system prompt holds the
instructions. Keep these short: they are paid for on every episode.
"""

from __future__ import annotations

from typing import Any

ENTITY_TYPES = ["Person", "Service", "Repository", "Team", "Organization", "Ticket", "Project", "Event", "Concept", "Other"]


def obj_schema(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _nullable(t: str) -> dict[str, Any]:
    return {"type": [t, "null"]}


# --------------------------------------------------------------------------- 1. extract entities

EXTRACT_ENTITIES_SYSTEM = f"""You build a company knowledge graph from chat messages (mostly Japanese).
Task: list the entities the CURRENT message talks about, and say whether it contains memorable facts.

Entity types: {", ".join(ENTITY_TYPES)}.
Rules:
- Extract people (use the name as written, without honorifics like さん), services/systems/repos/tools/software versions worth tracking, teams/orgs, tickets/PR numbers, projects, events/meetings, and important concepts/policies.
- Resolve pronouns and vague references (それ, あれ, this, 例の件) to the concrete entity using `context` (earlier messages) and `author`. Never output the pronoun itself.
- Use the most complete canonical name (e.g. "notify-bot", "Ruby on Rails"; a version of a product is the product: output "Rails", not "Rails 7.1", unless the version itself is the topic). Name PRs/issues with their repo or project ("iac-terraform PR #278"), never a bare number.
- Do NOT extract: the author (added automatically), greetings, generic nouns (タスク, 会議 without a specific name), dates, or entities only present in `context`.
- description: one short phrase about what the entity is, as far as this message tells.
- has_facts: true if the message states or implies something worth remembering later (who owns/does what, status, decision, plan, schedule, absence, dependency, problem, request assignment). false for pure chit-chat, thanks, acknowledgements, questions without new info."""

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

EXTRACT_FACTS_SYSTEM = """You extract temporal facts for a company knowledge graph from ONE chat message (mostly Japanese).
Input: `entities` (idx, name, type, current summary), `author`, `occurred_at` (local time, weekday, UTC offset), `context` (earlier messages, only for resolving references), `message`.

Output `facts`: relations between entities (subject idx -> object idx) or attributes of one entity (object null).
- Only facts stated or clearly implied by `message`. Do not repeat facts that only appear in `context` or in summaries.
- `fact`: one self-contained sentence in the message language that names its entities (no pronouns), e.g. "Bob は認証基盤の権限管理を担当している".
- `relation`: UPPER_SNAKE_CASE verb phrase (WORKS_ON, OWNS, MANAGES, DEPENDS_ON, HAS_STATUS, PLANS_TO, ABSENT, DECIDED, USES, HANDS_OVER_TO ...).
- valid_at: when the fact starts being true in the world, ISO 8601 (local time, no timezone, date-only is fine). Give a value ONLY if the message states or implies a start time (dates, 来週から, 明日, 9/14, 先月...), computed from `occurred_at` (assume the nearest sensible year). Otherwise null (= true from `occurred_at` on). Never guess a date.
- invalid_at: when the fact stops being true if the message says so (e.g. absence 9/14-9/16 -> 2026-09-17: the first moment it is no longer true), else null.
- modality: definite | planned (future intention/schedule) | tentative (unsure, proposal, question-like).
- A statement that changes an earlier state (new owner, new status) is a normal new fact; contradictions are handled later.
- Record durable knowledge: ownership/assignment, responsibility, status, decisions, plans/schedules, dependencies, where things are managed/stored, absences, problems and their causes. Do NOT record that someone asked / wonders / thanks / suggests looking something up, and skip chit-chat. Prefer 0-5 high-value facts.
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

# --------------------------------------------------------------------------- answer

ANSWER_SYSTEM = """あなたは社内の長期記憶(時系列ナレッジグラフ)を使って質問に答えるアシスタントです。
与えられた `facts`(と `entities`)だけを根拠に、日本語で簡潔に答えてください。
- 各主張の末尾に出典を [F1] のように付ける。必要なら送信者・日時も添える。
- 各 fact には status があります: valid=現在有効 / ended=すでに失効(過去の状態) / scheduled=まだ始まっていない予定。今の状態を聞かれたら valid を優先し、経緯を聞かれたら ended も使って時系列で説明する。
- valid_at / invalid_at (有効期間) を踏まえ、いつからいつまでの話かを明示する。`as_of` が指定されている場合はその時点の状況として答える。
- 情報が無い・不十分な場合は推測せず、分からないと述べる。矛盾や不確かさ (modality=tentative/planned) があれば明記する。"""

"""Value types returned by sqgraph: :class:`Node`, :class:`Relationship`, :class:`Path`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(eq=False)
class Node:
    """A graph node. Equality and hashing are by ``id``."""

    id: int
    labels: tuple[str, ...] = ()
    properties: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.properties[key]

    def __contains__(self, key: object) -> bool:
        return key in self.properties

    def get(self, key: str, default: Any = None) -> Any:
        """Return a property value or ``default``."""
        return self.properties.get(key, default)

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("node", self.id))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {"id": self.id, "labels": list(self.labels), "properties": dict(self.properties)}


@dataclass(eq=False)
class Relationship:
    """A directed, typed relationship. Equality and hashing are by ``id``."""

    id: int
    type: str
    start_node_id: int
    end_node_id: int
    properties: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.properties[key]

    def __contains__(self, key: object) -> bool:
        return key in self.properties

    def get(self, key: str, default: Any = None) -> Any:
        """Return a property value or ``default``."""
        return self.properties.get(key, default)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Relationship) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("rel", self.id))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "id": self.id,
            "type": self.type,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "properties": dict(self.properties),
        }


@dataclass(eq=False)
class Path:
    """An alternating sequence of nodes and relationships (returned for named paths)."""

    nodes: list[Node]
    relationships: list[Relationship]

    def __len__(self) -> int:
        return len(self.relationships)

    def __iter__(self) -> Iterator[Node | Relationship]:
        for i, n in enumerate(self.nodes):
            yield n
            if i < len(self.relationships):
                yield self.relationships[i]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Path)
            and other.nodes == self.nodes
            and other.relationships == self.relationships
        )

    def __hash__(self) -> int:
        return hash((tuple(n.id for n in self.nodes), tuple(r.id for r in self.relationships)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "relationships": [r.to_dict() for r in self.relationships],
        }

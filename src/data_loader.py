"""Dataset loading utilities for text-based knowledge graph experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

Triple = tuple[str, str, str]


@dataclass(slots=True)
class DatasetBundle:
    """Container for dataset splits, vocabularies, and optional text labels."""

    train: list[Triple]
    valid: list[Triple]
    test: list[Triple]
    entities: list[str]
    relations: list[str]
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    known_triples: set[Triple]
    entity_labels: dict[str, str]


def load_triples(file_path: str | Path) -> list[Triple]:
    """Load triples from a tab-separated file as a list of `(head, relation, tail)` tuples."""
    path = Path(file_path)
    triples: list[Triple] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected exactly 3 tab-separated fields in {path} on line {line_number}, got {len(parts)}."
                )

            triples.append((parts[0], parts[1], parts[2]))

    return preprocess_triples(triples)


def preprocess_triples(triples: Iterable[Triple]) -> list[Triple]:
    """Normalize whitespace and drop malformed triples before modeling."""
    processed: list[Triple] = []

    for head, relation, tail in triples:
        triple = (head.strip(), relation.strip(), tail.strip())
        if all(triple):
            processed.append(triple)

    return processed


def build_vocabularies(split_triples: Iterable[Iterable[Triple]]) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    """Build deterministic entity and relation vocabularies from dataset splits."""
    entity_set: set[str] = set()
    relation_set: set[str] = set()

    for triples in split_triples:
        for head, relation, tail in triples:
            entity_set.add(head)
            entity_set.add(tail)
            relation_set.add(relation)

    entities = sorted(entity_set)
    relations = sorted(relation_set)
    entity_to_id = {entity: index for index, entity in enumerate(entities)}
    relation_to_id = {relation: index for index, relation in enumerate(relations)}
    return entities, relations, entity_to_id, relation_to_id


def load_entity_labels(dataset_dir: str | Path) -> dict[str, str]:
    """Load optional entity surface-form labels when the dataset provides them."""
    path = Path(dataset_dir) / "entity2wikidata.json"
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, str] = {}

    for entity_id, metadata in payload.items():
        if not isinstance(metadata, dict):
            continue

        label = metadata.get("label")
        if isinstance(label, str) and label.strip():
            labels[entity_id] = label.strip()

    return labels


def _resolve_split_file(dataset_dir: Path, split_name: str) -> Path:
    """Resolve the on-disk file path for a dataset split."""
    candidate_names = {
        "train": ("train.txt", "train.tsv"),
        "valid": ("valid.txt", "validation.txt", "dev.txt", "valid.tsv"),
        "test": ("test.txt", "test.tsv"),
    }

    for candidate in candidate_names[split_name]:
        path = dataset_dir / candidate
        if path.exists():
            return path

    tried = ", ".join(candidate_names[split_name])
    raise FileNotFoundError(f"Could not find {split_name!r} split in {dataset_dir}. Tried: {tried}.")


def load_dataset(dataset_dir: str | Path) -> DatasetBundle:
    """Load dataset splits, vocabularies, and optional text labels from disk."""
    root = Path(dataset_dir)
    train = load_triples(_resolve_split_file(root, "train"))
    valid = load_triples(_resolve_split_file(root, "valid"))
    test = load_triples(_resolve_split_file(root, "test"))

    entities, relations, entity_to_id, relation_to_id = build_vocabularies((train, valid, test))

    return DatasetBundle(
        train=train,
        valid=valid,
        test=test,
        entities=entities,
        relations=relations,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        known_triples=set(train) | set(valid) | set(test),
        entity_labels=load_entity_labels(root),
    )

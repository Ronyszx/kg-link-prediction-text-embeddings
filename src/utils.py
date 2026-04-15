"""Shared utilities for path handling, seeding, and triple textualization."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

WORDNET_ENTITY_PATTERN = re.compile(r"^(?P<lemma>.+?)\.[a-z]\.\d+$")


def ensure_directory(path: str | Path) -> Path:
    """Normalize and create a directory path if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible experiments where possible."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_dataset_path(dataset_name: str, data_root: str | Path = "data") -> Path:
    """Resolve a dataset directory under the configured data root."""
    return Path(data_root) / dataset_name


def _split_camel_case(text: str) -> str:
    """Insert spaces between camelCase words without changing the tokens."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def clean_entity_text(entity: str, entity_labels: dict[str, str] | None = None) -> str:
    """Convert an entity identifier into a readable textual surface form."""
    if entity_labels and entity in entity_labels:
        return entity_labels[entity]

    text = entity.strip()
    match = WORDNET_ENTITY_PATTERN.match(text)
    if match:
        text = match.group("lemma")

    text = _split_camel_case(text)
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def relation_to_text(relation: str) -> str:
    """Convert a relation identifier into a readable phrase."""
    text = relation.strip().lstrip("_")

    if "/" in text:
        segments = [segment for segment in text.split("/") if segment]
        text = segments[-1] if segments else text

    if "." in text:
        segments = [segment for segment in text.split(".") if segment]
        text = segments[-1] if segments else text

    text = _split_camel_case(text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compose_statement(head_text: str, relation_text: str, tail_text: str) -> str:
    """Compose a lightweight natural-language sentence from triple components."""
    if relation_text.startswith(("is ", "are ", "was ", "were ", "has ", "have ", "plays ", "died ", "lives ")):
        return f"{head_text} {relation_text} {tail_text}"

    if relation_text.endswith(" of"):
        return f"{head_text} is the {relation_text} {tail_text}"

    return f"{head_text} has relation {relation_text} with {tail_text}"


def triple_to_text(
    head: str,
    relation: str,
    tail: str,
    entity_labels: dict[str, str] | None = None,
) -> str:
    """Convert a symbolic triple into a readable sentence for embedding."""
    head_text = clean_entity_text(head, entity_labels=entity_labels)
    relation_text = relation_to_text(relation)
    tail_text = clean_entity_text(tail, entity_labels=entity_labels)
    return _compose_statement(head_text, relation_text, tail_text)


def masked_triple_to_text(
    head: str | None,
    relation: str,
    tail: str | None,
    missing: str,
    entity_labels: dict[str, str] | None = None,
    mask_token: str = "[MASK]",
) -> str:
    """Create a natural-language query sentence with a missing head or tail."""
    if missing == "head":
        if tail is None:
            raise ValueError("Tail must be provided when masking the head.")
        return _compose_statement(mask_token, relation_to_text(relation), clean_entity_text(tail, entity_labels=entity_labels))

    if missing == "tail":
        if head is None:
            raise ValueError("Head must be provided when masking the tail.")
        return _compose_statement(clean_entity_text(head, entity_labels=entity_labels), relation_to_text(relation), mask_token)

    raise ValueError(f"Unsupported missing target: {missing!r}. Expected 'head' or 'tail'.")

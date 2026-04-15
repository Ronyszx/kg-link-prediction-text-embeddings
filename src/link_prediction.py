"""Link prediction with a text bi-encoder over natural-language triples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import Triple
from src.text_encoder import TextEncoder
from src.utils import masked_triple_to_text, triple_to_text


@dataclass(slots=True)
class RankedPredictions:
    """Sorted candidate ranking returned by a prediction query."""

    query_text: str
    ranking: pd.DataFrame


class TextEmbeddingLinkPredictor:
    """Rank candidate heads or tails using similarities in embedding space.

    This baseline treats knowledge graph facts as short natural-language documents.
    Compared with dedicated KG embedding models, it can exploit semantics from entity
    and relation names without learning task-specific embeddings from scratch.

    The trade-off is that this approach does not model graph structure explicitly.
    It is also lighter than cross-encoder approaches such as KG-BERT at inference
    time, but it usually sacrifices some interaction fidelity because query and
    candidate triples are encoded independently.
    """

    def __init__(self, encoder: TextEncoder, entity_labels: dict[str, str] | None = None) -> None:
        """Initialize the predictor with a shared text encoder and label mapping."""
        self.encoder = encoder
        self.entity_labels = entity_labels or {}

    def _rank_candidates(self, query_text: str, candidate_texts: Sequence[str], candidate_entities: Sequence[str]) -> RankedPredictions:
        """Embed the query and candidates, then return a descending similarity ranking."""
        if not candidate_entities:
            raise ValueError("Candidate entity list must not be empty.")

        query_embedding = self.encoder.encode_queries([query_text])[0]
        candidate_embeddings = self.encoder.encode_documents(candidate_texts)

        # Exact ranking over all entities is simple and faithful to the baseline,
        # but it becomes expensive for large graphs. For full-scale experiments,
        # precomputed ANN indices or relation-specific candidate pruning are useful.
        scores = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).ravel()
        sorted_indices = np.argsort(-scores)

        ranking = pd.DataFrame(
            {
                "candidate": [candidate_entities[index] for index in sorted_indices],
                "score": scores[sorted_indices],
                "candidate_text": [candidate_texts[index] for index in sorted_indices],
            }
        )
        return RankedPredictions(query_text=query_text, ranking=ranking)

    def predict_tail(self, head: str, relation: str, all_entities: Sequence[str]) -> RankedPredictions:
        """Rank candidate tail entities for an incomplete `(head, relation, ?)` query."""
        query_text = masked_triple_to_text(
            head=head,
            relation=relation,
            tail=None,
            missing="tail",
            entity_labels=self.entity_labels,
        )
        candidate_texts = [
            triple_to_text(head, relation, candidate_tail, entity_labels=self.entity_labels)
            for candidate_tail in all_entities
        ]
        return self._rank_candidates(query_text, candidate_texts, all_entities)

    def predict_head(self, relation: str, tail: str, all_entities: Sequence[str]) -> RankedPredictions:
        """Rank candidate head entities for an incomplete `(?, relation, tail)` query."""
        query_text = masked_triple_to_text(
            head=None,
            relation=relation,
            tail=tail,
            missing="head",
            entity_labels=self.entity_labels,
        )
        candidate_texts = [
            triple_to_text(candidate_head, relation, tail, entity_labels=self.entity_labels)
            for candidate_head in all_entities
        ]
        return self._rank_candidates(query_text, candidate_texts, all_entities)


def build_candidate_space(triples: Sequence[Triple]) -> dict[str, list[str]]:
    """Construct entity and relation candidate spaces from a collection of triples."""
    entities = sorted({entity for head, _, tail in triples for entity in (head, tail)})
    relations = sorted({relation for _, relation, _ in triples})
    return {"entities": entities, "relations": relations}


def score_triples(triples: Sequence[Triple], encoder: TextEncoder, entity_labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Embed fully observed triples and return a simple scored view for inspection."""
    texts = [triple_to_text(head, relation, tail, entity_labels=entity_labels) for head, relation, tail in triples]
    embeddings = encoder.encode_documents(texts)
    norms = np.linalg.norm(embeddings, axis=1)
    return pd.DataFrame({"triple_text": texts, "embedding_norm": norms})


def predict_missing_links(
    predictor: TextEmbeddingLinkPredictor,
    query_triples: Sequence[Triple],
    all_entities: Sequence[str],
) -> pd.DataFrame:
    """Run both head and tail prediction for a batch of query triples."""
    records: list[dict[str, object]] = []

    for head, relation, tail in query_triples:
        tail_predictions = predictor.predict_tail(head, relation, all_entities)
        head_predictions = predictor.predict_head(relation, tail, all_entities)
        records.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "top_tail_prediction": tail_predictions.ranking.iloc[0]["candidate"],
                "top_head_prediction": head_predictions.ranking.iloc[0]["candidate"],
            }
        )

    return pd.DataFrame(records)

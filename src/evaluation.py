"""Ranking metrics and evaluation loops for link prediction experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import Triple
from src.link_prediction import RankedPredictions, TextEmbeddingLinkPredictor


def mean_rank(ranks: Iterable[int]) -> float:
    """Compute the arithmetic mean of integer ranks."""
    rank_array = np.asarray(list(ranks), dtype=np.float32)
    return float(rank_array.mean()) if rank_array.size else 0.0


def mean_reciprocal_rank(ranks: Iterable[int]) -> float:
    """Compute the average reciprocal rank for a set of predictions."""
    rank_array = np.asarray(list(ranks), dtype=np.float32)
    return float((1.0 / rank_array).mean()) if rank_array.size else 0.0


def hits_at_k(ranks: Iterable[int], k: int) -> float:
    """Compute the proportion of ranks that fall within the top-k positions."""
    rank_array = np.asarray(list(ranks), dtype=np.int32)
    return float((rank_array <= k).mean()) if rank_array.size else 0.0


def build_filter_mappings(known_triples: Iterable[Triple]) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]]]:
    """Index known triples for filtered ranking evaluation."""
    tails_by_head_relation: dict[tuple[str, str], set[str]] = defaultdict(set)
    heads_by_relation_tail: dict[tuple[str, str], set[str]] = defaultdict(set)

    for head, relation, tail in known_triples:
        tails_by_head_relation[(head, relation)].add(tail)
        heads_by_relation_tail[(relation, tail)].add(head)

    return dict(tails_by_head_relation), dict(heads_by_relation_tail)


def rank_target_prediction(predictions: RankedPredictions, target_entity: str, filtered_entities: set[str] | None = None) -> int:
    """Return the 1-based rank of the gold entity after optional filtered evaluation."""
    ranking = predictions.ranking
    if filtered_entities:
        ranking = ranking.loc[~ranking["candidate"].isin(filtered_entities)].reset_index(drop=True)

    matches = ranking.index[ranking["candidate"] == target_entity]
    if len(matches) == 0:
        raise ValueError(f"Target entity {target_entity!r} was not found in the candidate ranking.")

    return int(matches[0] + 1)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Aggregate MR, MRR, and Hits@K from a prediction results table."""
    if "rank" in predictions.columns:
        ranks = predictions["rank"].tolist()
    else:
        ranks: list[int] = []
        for column in ("head_rank", "tail_rank"):
            if column in predictions.columns:
                ranks.extend(predictions[column].tolist())

    return {
        "MR": mean_rank(ranks),
        "MRR": mean_reciprocal_rank(ranks),
        "Hits@1": hits_at_k(ranks, 1),
        "Hits@3": hits_at_k(ranks, 3),
        "Hits@5": hits_at_k(ranks, 5),
        "Hits@10": hits_at_k(ranks, 10),
    }


def evaluate_link_prediction(
    predictor: TextEmbeddingLinkPredictor,
    triples: Sequence[Triple],
    all_entities: Sequence[str],
    known_triples: Iterable[Triple],
    limit: int = 100,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate head and tail prediction on a small subset of triples.

    The default subset keeps the entry point lightweight enough for debugging.
    Full datasets require more aggressive optimization because exact ranking
    still scales linearly with the number of candidate entities per query.
    """

    tails_by_head_relation, heads_by_relation_tail = build_filter_mappings(known_triples)
    subset = list(triples[:limit])
    iterator = tqdm(subset, desc="Evaluating triples", disable=not show_progress)

    records: list[dict[str, object]] = []

    for head, relation, tail in iterator:
        tail_predictions = predictor.predict_tail(head, relation, all_entities)
        head_predictions = predictor.predict_head(relation, tail, all_entities)

        tail_filtered = set(tails_by_head_relation.get((head, relation), set()))
        tail_filtered.discard(tail)

        head_filtered = set(heads_by_relation_tail.get((relation, tail), set()))
        head_filtered.discard(head)

        tail_rank = rank_target_prediction(tail_predictions, tail, tail_filtered)
        head_rank = rank_target_prediction(head_predictions, head, head_filtered)

        records.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "tail_rank": tail_rank,
                "head_rank": head_rank,
                "top_tail_prediction": tail_predictions.ranking.iloc[0]["candidate"],
                "top_head_prediction": head_predictions.ranking.iloc[0]["candidate"],
                "tail_query_text": tail_predictions.query_text,
                "head_query_text": head_predictions.query_text,
            }
        )

    prediction_frame = pd.DataFrame(records)
    return prediction_frame, evaluate_predictions(prediction_frame)

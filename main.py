"""Entry point for text-embedding-based knowledge graph link prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import DatasetBundle, load_dataset
from src.evaluation import evaluate_link_prediction
from src.link_prediction import TextEmbeddingLinkPredictor
from src.text_encoder import TextEncoder
from src.utils import get_dataset_path, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse lightweight experiment arguments for the baseline pipeline."""
    parser = argparse.ArgumentParser(description="Knowledge graph link prediction with text embeddings.")
    parser.add_argument("--dataset", default="FB15K", help="Dataset directory name under the data root.")
    parser.add_argument("--data-root", default="data", help="Root directory containing datasets.")
    parser.add_argument("--subset-size", type=int, default=100, help="Number of triples to evaluate for debugging.")
    parser.add_argument(
        "--model-name",
        default="nomic-ai/nomic-embed-text-v1.5",
        help="SentenceTransformer model name used for text embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def print_dataset_summary(dataset_name: str, dataset: DatasetBundle, subset_size: int) -> None:
    """Print a compact summary before running evaluation."""
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Train triples: {len(dataset.train):,}")
    print(f"Validation triples: {len(dataset.valid):,}")
    print(f"Test triples: {len(dataset.test):,}")
    print(f"Entities: {len(dataset.entities):,}")
    print(f"Relations: {len(dataset.relations):,}")
    print(f"Evaluation subset size: {min(subset_size, len(dataset.valid)):,}")
    print("=" * 80)


def print_metrics(metrics: dict[str, float]) -> None:
    """Print evaluation metrics in a clear, research-report-friendly format."""
    print("\nEvaluation Metrics")
    print("-" * 80)
    for key in ("MR", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"):
        print(f"{key:<8}: {metrics[key]:.4f}")
    print("-" * 80)


def run_pipeline(
    dataset_name: str = "FB15K",
    data_root: str | Path = "data",
    subset_size: int = 100,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    batch_size: int = 128,
) -> None:
    """Run a small-scale link prediction experiment on the requested dataset.

    Text embeddings provide a strong semantic prior when entity and relation names
    are meaningful or when external labels are available. This makes the setup
    attractive for research prototypes and zero-shot-style exploration.

    The main limitation is structural: unlike dedicated graph models, this baseline
    does not directly encode multi-hop topology or neighborhood statistics. Exact
    ranking over every entity is also computationally expensive, so the pipeline
    intentionally evaluates a small validation subset first.
    """

    dataset_path = get_dataset_path(dataset_name, data_root=data_root)
    dataset = load_dataset(dataset_path)
    print_dataset_summary(dataset_name, dataset, subset_size)

    # We evaluate on the validation split by default because this is a research
    # scaffold, and keeping test data untouched is usually the safer workflow.
    evaluation_split = dataset.valid

    # A bi-encoder is cheaper than cross-encoder scoring at ranking time because
    # queries and candidates are encoded independently. The trade-off is weaker
    # interaction modeling than methods such as KG-BERT or graph-specific models.
    encoder = TextEncoder(model_name=model_name, batch_size=batch_size)
    predictor = TextEmbeddingLinkPredictor(encoder=encoder, entity_labels=dataset.entity_labels)

    predictions, metrics = evaluate_link_prediction(
        predictor=predictor,
        triples=evaluation_split,
        all_entities=dataset.entities,
        known_triples=dataset.known_triples,
        limit=subset_size,
        show_progress=True,
    )

    print_metrics(metrics)

    if not predictions.empty:
        print("\nSample Predictions")
        print("-" * 80)
        preview = predictions.head(5)[
            ["head", "relation", "tail", "top_head_prediction", "top_tail_prediction", "head_rank", "tail_rank"]
        ]
        print(preview.to_string(index=False))
        print("-" * 80)


def main() -> None:
    """Parse arguments, seed the environment, and launch the baseline experiment."""
    args = parse_args()
    seed_everything(args.seed)
    run_pipeline(
        dataset_name=args.dataset,
        data_root=args.data_root,
        subset_size=args.subset_size,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

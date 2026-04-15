# Knowledge Graph Link Prediction with Text Embeddings

## Project Overview

This repository provides a modular, research-oriented baseline for knowledge graph link prediction using natural-language representations of triples and sentence embeddings. The current implementation uses `nomic-ai/nomic-embed-text-v1.5` through `sentence-transformers`, converts triples into readable text, and ranks candidate entities with cosine similarity.

The code is intentionally organized like a small research project:

- `src/data_loader.py` handles dataset parsing and vocabulary construction.
- `src/text_encoder.py` wraps the embedding model and caches repeated texts.
- `src/link_prediction.py` performs head and tail ranking.
- `src/evaluation.py` computes MR, MRR, and Hits@K.
- `src/utils.py` contains shared utilities for seeding, path handling, and triple textualization.

## Setup Instructions

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your datasets under the `data/` directory using the expected structure shown below.

## How to Run the Code

Run a small validation subset on FB15K:

```bash
python3 main.py --dataset FB15K --subset-size 100
```

Useful options:

```bash
python3 main.py --dataset WN18RR --subset-size 50
python3 main.py --dataset YAGO3-10 --subset-size 25 --batch-size 64
```

Notes:

- The loader expects tab-separated triples in `train.txt`, `valid.txt`, and `test.txt`.
- Evaluation currently uses a small validation subset for debugging because exact ranking over all entities is computationally expensive.
- FB15K label metadata in `entity2wikidata.json` is used automatically when available so text embeddings see readable entity names instead of opaque IDs.

## Expected Folder Structure for Datasets

```text
project_root/
├── data/
│   ├── FB15K/
│   ├── WN18RR/
│   └── YAGO3-10/
├── src/
│   ├── data_loader.py
│   ├── text_encoder.py
│   ├── link_prediction.py
│   ├── evaluation.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

Expected dataset layout:

```text
data/
├── FB15K/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── entity2wikidata.json   # optional but useful for readable FB15K entity text
├── WN18RR/
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
└── YAGO3-10/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

Each line should contain one triple in the format:

```text
head<TAB>relation<TAB>tail
```

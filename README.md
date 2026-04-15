# Knowledge Graph Link Prediction with Text Embeddings

## Project Overview

This project is a simple baseline for knowledge graph link prediction using text embeddings instead of traditional KGE models. The main idea is to convert triples into natural language sentences and use a pretrained embedding model to rank possible candidates.

I used the `nomic-ai/nomic-embed-text-v1.5` model (via `sentence-transformers`) and treated link prediction as a similarity problem. Given a partially missing triple, the model scores all possible candidates and ranks them based on cosine similarity.

The project is structured in a modular way so it’s easier to experiment and extend:

* `src/data_loader.py` → loads datasets and builds entity/relation vocab
* `src/text_encoder.py` → handles embedding + simple caching (to avoid recomputing)
* `src/link_prediction.py` → does head/tail prediction and ranking
* `src/evaluation.py` → computes MR, MRR, Hits@K (filtered setting)
* `src/utils.py` → helper functions (text conversion, seeding, etc.)

---

## Setup Instructions

1. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Example (FB15K with small subset):

```bash
python3 main.py --dataset FB15K --subset-size 100
```

You can also try:

```bash
python3 main.py --dataset WN18RR --subset-size 50
python3 main.py --dataset YAGO3-10 --subset-size 25 --batch-size 64
```

---

## Notes

* The dataset should be in tab-separated format (`head relation tail`)
* I’m using only a small subset for evaluation because ranking against all entities (~15k) is quite slow
* FB15K works better if `entity2wikidata.json` is available since IDs get converted into readable text
* This is a **baseline**, so performance is not very high (expected)

---

## Dataset Setup

Download datasets from:
https://github.com/villmow/datasets_knowledge_embedding

Place them like this:

```text
project_root/
├── data/
│   ├── FB15K/
│   ├── WN18RR/
│   └── YAGO3-10/
```

Expected structure:

```text
data/
├── FB15K/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── entity2wikidata.json   # optional but helpful
├── WN18RR/
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
└── YAGO3-10/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

Each line in the files should look like:

```text
head<TAB>relation<TAB>tail
```

---

## Final Thoughts

This approach works by leveraging semantic similarity, but it doesn’t capture graph structure like traditional methods (e.g., TransE or KG-BERT). So the results are limited, but it’s a good starting point for experimenting with text-based approaches.

If I had more time, I would:

* precompute entity embeddings for speed
* try better sentence templates
* maybe combine this with graph-based methods

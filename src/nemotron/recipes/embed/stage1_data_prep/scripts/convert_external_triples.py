#!/usr/bin/env python3
"""Convert external query/positive/negative data into the Stage 2 finetune format.

Use this when you ALREADY have triples (query, positive passage, hard negatives)
and want to skip Stage 0 (synthetic data generation) and Stage 1 (hard negative
mining) entirely, feeding the data directly to `nemotron embed finetune`.

Input
-----
A JSON or JSONL file. Each example may use any of these field spellings:

    {
        "query":     "the question text",          # or "question"
        "positive":  "the relevant passage",       # or "positives": ["...", ...]
        "negatives": ["irrelevant passage", ...]   # or "negative": "..."
    }

Output (written into --output-dir)
----------------------------------
    train.json                 # automodel format consumed by Stage 2
    corpus/train.parquet       # id -> text table referenced by train.json
    corpus/merlin_metadata.json

Then run:
    nemotron embed finetune -c default \
        train_data_path=/abs/path/to/<output-dir>/train.json

Usage
-----
    python convert_external_triples.py my_data.jsonl --output-dir ./output/embed/external --corpus-id my_corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _doc_id(text: str) -> str:
    """Deterministic content-addressed id so identical passages dedupe to one row."""
    return "d_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Try a single JSON array first, then fall back to JSONL.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "data" in parsed:
            return list(parsed["data"])
        return [parsed]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _get_query(record: dict[str, Any]) -> str:
    for key in ("query", "question", "anchor"):
        if record.get(key):
            return str(record[key])
    return ""


def _get_positives(record: dict[str, Any]) -> list[str]:
    if record.get("positives"):
        vals = record["positives"]
        return [str(v) for v in vals] if isinstance(vals, list) else [str(vals)]
    for key in ("positive", "pos", "pos_doc"):
        if record.get(key):
            return [str(record[key])]
    return []


def _get_negatives(record: dict[str, Any]) -> list[str]:
    for key in ("negatives", "neg", "neg_doc", "hard_negatives"):
        if record.get(key):
            vals = record[key]
            return [str(v) for v in vals] if isinstance(vals, list) else [str(vals)]
    if record.get("negative"):
        return [str(record["negative"])]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=str, help="Path to JSON/JSONL file with query/positive/negative triples.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write train.json + corpus/.")
    parser.add_argument("--corpus-id", type=str, default="my_corpus", help="Corpus identifier (label only).")
    parser.add_argument(
        "--min-negatives",
        type=int,
        default=0,
        help="Drop examples with fewer than this many negatives (set to train_n_passages-1 to avoid silent skips in Stage 2).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(input_path)
    print(f"Loaded {len(records)} raw records from {input_path}")

    corpus: dict[str, str] = {}  # text -> id
    data: list[dict[str, Any]] = []
    skipped = 0

    for idx, record in enumerate(records):
        query = _get_query(record).strip()
        positives = [p.strip() for p in _get_positives(record) if p and p.strip()]
        negatives = [n.strip() for n in _get_negatives(record) if n and n.strip()]

        if not query or not positives:
            skipped += 1
            continue
        if len(negatives) < args.min_negatives:
            skipped += 1
            continue

        pos_doc = []
        for text in positives:
            doc_id = corpus.setdefault(text, _doc_id(text))
            pos_doc.append({"id": doc_id})

        neg_doc = []
        for text in negatives:
            doc_id = corpus.setdefault(text, _doc_id(text))
            neg_doc.append({"id": doc_id})

        data.append(
            {
                "question_id": f"q{idx}",
                "question": query,
                "corpus_id": args.corpus_id,
                "pos_doc": pos_doc,
                "neg_doc": neg_doc,
            }
        )

    if not data:
        print("Error: no usable examples after parsing. Check field names in your input.")
        return 1

    # Write corpus parquet (id, text).
    corpus_rows = [{"id": doc_id, "text": text} for text, doc_id in corpus.items()]
    pd.DataFrame(corpus_rows).to_parquet(corpus_dir / "train.parquet", index=False)

    with open(corpus_dir / "merlin_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"corpus_id": args.corpus_id, "class": "TextQADataset"}, f, indent=2)

    train_json_path = output_dir / "train.json"
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump({"corpus": {"path": "./corpus/"}, "data": data}, f, indent=2)

    neg_counts = sorted(len(r["neg_doc"]) for r in data)
    median_neg = neg_counts[len(neg_counts) // 2]
    print(f"Wrote {train_json_path}")
    print(f"  Examples:           {len(data)} (skipped {skipped})")
    print(f"  Unique passages:    {len(corpus_rows)} -> corpus/train.parquet")
    print(f"  Median negatives:   {median_neg}")
    print()
    print("Next:")
    print(f"  nemotron embed finetune -c default train_data_path={train_json_path}")
    if median_neg < 4:
        print(f"  (median negatives={median_neg} < 4: also set train_n_passages={median_neg + 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

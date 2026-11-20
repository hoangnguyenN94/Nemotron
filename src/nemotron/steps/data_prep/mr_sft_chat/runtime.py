# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chat-format SFT rows from one language subset of Tiny Aya multilingual reasoning.

Each dataset row holds a question, a thinking trace and an answer in the subset's
language, plus the English originals of all three. A row therefore gives two
training examples: one in the subset language and one in English. Examples that
would not fit the training sequence length are dropped, because the packing step
would otherwise cut off the end of the answer.
"""

from __future__ import annotations

import collections
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TARGET_FIELDS = ("question", "thinking", "output")
ENGLISH_FIELDS = ("original_question", "original_thinking", "original_output")


def chat_record(row: Mapping[str, Any], fields: tuple[str, str, str], language: str) -> dict[str, Any] | None:
    """One user turn and one assistant turn with its reasoning, or None when a part is empty."""
    question, thinking, answer = (str(row.get(field) or "").strip() for field in fields)
    if not (question and thinking and answer):
        return None
    return {
        "id": f"{row['id']}:{language}",
        "language": language,
        "domain": row.get("domain"),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "reasoning_content": thinking, "content": answer},
        ],
    }


def build_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    language: str,
    include_english: bool = True,
    count_tokens: Callable[[list[dict[str, Any]]], int] | None = None,
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Chat records for ``rows`` and, per language, how many were kept, empty or too long."""
    variants = [(language, TARGET_FIELDS)] + ([("en", ENGLISH_FIELDS)] if include_english else [])
    records: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()
    for row in rows:
        for lang, fields in variants:
            record = chat_record(row, fields, lang)
            if record is None:
                stats[f"{lang}_empty"] += 1
            elif max_tokens and count_tokens and count_tokens(record["messages"]) > max_tokens:
                stats[f"{lang}_too_long"] += 1
            else:
                stats[f"{lang}_kept"] += 1
                records.append(record)
    return records, dict(stats)


def token_counter(tokenizer_name: str) -> Callable[[list[dict[str, Any]]], int]:
    """Token count of a conversation rendered with the tokenizer's own chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(messages: list[dict[str, Any]]) -> int:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def run_mr_sft_chat(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``<split>.jsonl``, a packing blend per split and ``stats.json`` under ``output_dir``."""
    from datasets import load_dataset

    output_dir = Path(str(cfg["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    language = str(cfg["subset"])
    max_tokens = cfg.get("max_tokens")
    count_tokens = token_counter(str(cfg["tokenizer"])) if max_tokens else None

    stats: dict[str, Any] = {
        "dataset": cfg["dataset"],
        "subset": language,
        "include_english": bool(cfg.get("include_english", True)),
        "tokenizer": cfg.get("tokenizer"),
        "max_tokens": max_tokens,
        "splits": {},
    }
    for name, hf_split in dict(cfg["splits"]).items():
        rows = load_dataset(str(cfg["dataset"]), language, split=str(hf_split))
        if cfg.get("max_rows"):
            rows = rows.select(range(min(int(cfg["max_rows"]), len(rows))))
        records, counts = build_records(
            rows,
            language=language,
            include_english=stats["include_english"],
            count_tokens=count_tokens,
            max_tokens=max_tokens,
        )
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        blend = {"datasets": [{"name": f"mr_sft_{language}_{name}", "path": str(path)}]}
        (output_dir / f"blend_{name}.json").write_text(json.dumps(blend, indent=2) + "\n", encoding="utf-8")
        stats["splits"][name] = {"source_split": hf_split, "rows": len(rows), "examples": len(records), **counts}
        logger.info("%s: %d rows -> %d examples %s -> %s", name, len(rows), len(records), counts, path)

    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats

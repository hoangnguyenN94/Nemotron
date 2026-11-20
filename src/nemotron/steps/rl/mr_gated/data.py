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

"""Training rows for gated GRPO: GSM8K in Vietnamese plus a scope-preserving share of English.

Every row satisfies the gated prompt contract in ``schema.py``: the prompt is
the plain question and the row carries a label, never a gold field. Labels are
the datasets' reference answers.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from nemotron.steps.rl.mr_gated.schema import GATED_ARM, validate_gated_record

VI_DATASET = "namfam/gsm8k-vietnamese"
EN_DATASET = "openai/gsm8k"
VI_SOURCE = "gsm8k_train_vi"
EN_SOURCE = "gsm8k_train_en"

_CALC = re.compile(r"<<([^>=]*)=")


def scope_key(solution: str) -> tuple[int, str]:
    """Number of calculation steps and the arithmetic operations in a GSM8K reference solution."""
    calcs = _CALC.findall(solution)
    return len(calcs), "".join(sorted(set(re.findall(r"[-+*/]", "".join(calcs)))))


def spread_subset(items: Sequence[Any], fraction: float, key: Callable[[Any], Any]) -> list[Any]:
    """Evenly spaced picks from ``items`` sorted by ``key``.

    Sorting by scope first means every kind of item keeps its share of the
    subset, and the picks inside one kind cover its whole range.
    """
    ordered = sorted(items, key=key)
    count = int(len(ordered) * fraction)
    return [ordered[int((i + 0.5) * len(ordered) / count)] for i in range(count)]


def gated_row(source: str, index: int, split: str, lang: str, question: str, label: str) -> dict[str, Any]:
    question = question.strip()
    return {
        "item_id": f"{source}:{index}",
        "source": source,
        "split": split,
        "task": "numeric",
        "prompt_language": lang,
        "question": question,
        "label": label.strip().replace(",", ""),
        "unassisted_user": question,
        "path": "train",
        "arm": GATED_ARM,
    }


def build_rows(split: str = "train", english_fraction: float = 0.25, seed: int = 42) -> list[dict[str, Any]]:
    """All Vietnamese questions of ``split`` plus ``english_fraction`` of their English originals, shuffled.

    The Vietnamese ``train`` and ``val`` splits partition the GSM8K train set by
    its ``index`` column, so English rows are drawn only from the questions of
    the same split. The GSM8K test split is never loaded.
    """
    from datasets import load_dataset

    vietnamese = load_dataset(VI_DATASET)[split]
    rows = [gated_row(VI_SOURCE, row["index"], split, "vi", row["problem"], str(row["answer"])) for row in vietnamese]
    if english_fraction > 0:
        english = load_dataset(EN_DATASET, "main")["train"]
        paired = [(index, english[index]) for index in vietnamese["index"]]
        chosen = spread_subset(
            paired,
            english_fraction,
            key=lambda pair: (*scope_key(pair[1]["answer"]), len(pair[1]["question"]), pair[0]),
        )
        rows += [
            gated_row(EN_SOURCE, index, split, "en", row["question"], row["answer"].split("####")[1])
            for index, row in chosen
        ]
    random.Random(seed).shuffle(rows)
    for row in rows:
        validate_gated_record(row)
    return rows

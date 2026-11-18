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

"""Gated RL prompt contract: unassisted question + frozen label, never gold."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from nemotron.steps.eval.mr_l2.constants import EVAL_SOURCE_NAMES, GOLD_FIELDS
from nemotron.steps.eval.mr_l2.contamination import ContaminationError

GATED_ARM = "approach1_gated"
HINT_HEADER = "Gợi ý:"
GATED_FIELDS = (
    "item_id",
    "source",
    "split",
    "task",
    "prompt_language",
    "question",
    "label",
    "unassisted_user",
    "path",
    "arm",
)
_FORBIDDEN_KEYS = (
    "hint",
    "guided_user",
    "hint_strategy",
    "hint_source_seed",
    "messages",
    "reasoning_content",
)


class TrainSchemaError(ValueError):
    """Gated prompt JSONL violated the train-path contract."""


def gold_fields_present(record: Mapping[str, object]) -> tuple[str, ...]:
    found = []
    for key in GOLD_FIELDS:
        if key not in record:
            continue
        value = record[key]
        if value in (None, ""):
            continue
        found.append(key)
    return tuple(found)


def refuse_gold(record: Mapping[str, Any], *, where: str) -> None:
    leaked = gold_fields_present(record)
    if leaked:
        raise ContaminationError(f"gold fields on {where}: {leaked}")


def validate_gated_record(
    record: Mapping[str, object],
    eval_ids: Iterable[str] = (),
) -> None:
    refuse_gold(record, where="gated record")
    item_id = str(record.get("item_id") or "")
    blocked = set(eval_ids)
    if item_id and item_id in blocked:
        raise ContaminationError(f"eval-bench id on train path: {item_id}")
    source = str(record.get("source") or "")
    if source in set(EVAL_SOURCE_NAMES):
        raise ContaminationError(f"eval source on train path: {source}")
    if str(record.get("path") or "") != "train":
        raise TrainSchemaError("gated record path must be 'train'")
    if str(record.get("arm") or "") != GATED_ARM:
        raise TrainSchemaError(f"gated record arm must be {GATED_ARM}")
    missing = [key for key in GATED_FIELDS if key not in record]
    if missing:
        raise TrainSchemaError(f"missing fields: {missing}")
    question = str(record.get("question") or "").rstrip()
    user = str(record.get("unassisted_user") or "")
    if user != question:
        raise TrainSchemaError("unassisted_user must equal the plain question")
    if HINT_HEADER in user:
        raise TrainSchemaError("hint header leaked into unassisted_user")
    for key in _FORBIDDEN_KEYS:
        value = record.get(key)
        if value not in (None, "", []):
            raise TrainSchemaError(f"gated record must not carry {key}")
    if not str(record.get("label") or "").strip():
        raise TrainSchemaError("gated record missing label")
    if not question:
        raise TrainSchemaError("gated record missing question")

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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemotron.steps.eval.mr_l2.constants import SAMPLE_LOG_FIELDS
from nemotron.steps.eval.mr_l2.types import EvalMode, PathKind, PromptRequest, ScoreResult
from nemotron.steps.eval.mr_bench.prompts import sampling_hash


def sample_log(
    *,
    item_id: str,
    source: str,
    split: str | None,
    mode: EvalMode,
    path: PathKind,
    task: str,
    prompt_language: str,
    prompt: PromptRequest,
    raw_output: str,
    score: ScoreResult,
    gold: str | None = None,
) -> dict[str, Any]:
    gold_out = gold if path == "eval" else None
    correct = score.correct if path == "eval" else None
    record = {
        "item_id": item_id,
        "source": source,
        "split": split,
        "mode": mode,
        "path": path,
        "task": task,
        "prompt_language": prompt_language,
        "requested_answer_language": prompt.requested_answer_language,
        "thinking_language_target": prompt.thinking_language_target,
        "user_message": prompt.user_message,
        "thinking_prefix": prompt.thinking_prefix,
        "raw_output": raw_output,
        "thinking": score.split.thinking,
        "answer_span": score.split.answer_span,
        "parsed_answer": score.parsed_answer,
        "parser_answer": score.parser_answer,
        "parser_correct": score.parser_correct if path == "eval" else None,
        "thinking_lang_share": score.thinking_lang_share,
        "in_language_95": score.in_language_95,
        "answer_language": score.answer_language,
        "closed": score.closed,
        "truncated": score.truncated,
        "loop": score.loop,
        "n_tokens": score.n_tokens,
        "sampling_hash": sampling_hash(prompt.sampling),
        "correct": correct,
        "gold": gold_out,
        "hard_gate": score.hard_gate,
    }
    return record


def validate_log_record(record: Mapping[str, object], *, path: PathKind | None = None) -> None:
    missing = [k for k in SAMPLE_LOG_FIELDS if k not in record]
    if missing:
        raise ValueError(f"log record missing fields: {missing}")
    kind = path or str(record.get("path") or "")
    if kind == "train" and record.get("gold") not in (None, ""):
        raise ValueError("train log must not carry gold")
    if kind == "train" and record.get("correct") not in (None, ""):
        raise ValueError("train log must not carry correct")

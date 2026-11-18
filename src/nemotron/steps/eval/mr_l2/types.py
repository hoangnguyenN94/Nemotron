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

from dataclasses import dataclass, field
from typing import Literal, Sequence

PathKind = Literal["eval", "train"]
EvalMode = Literal["free", "forced"]
TaskKind = Literal["numeric", "mcq_aj", "mcq_ae", "polymath", "instruction"]


@dataclass(frozen=True)
class ThinkAnswerSplit:
    thinking: str
    answer_span: str
    think_open_present: bool
    think_close_present: bool


@dataclass(frozen=True)
class ClosureResult:
    closed: bool
    truncated: bool
    loop: bool
    think_close_present: bool


@dataclass(frozen=True)
class LanguageScore:
    share: float
    in_language_95: bool
    n_sentences: int
    n_scored: int
    n_allowed: int
    languages: tuple[str | None, ...]


@dataclass(frozen=True)
class ScoreResult:
    split: ThinkAnswerSplit
    parsed_answer: str
    thinking_lang_share: float
    in_language_95: bool
    thinking_lang_ok: bool
    answer_language: str
    answer_lang_ok: bool
    closed: bool
    truncated: bool
    loop: bool
    correct: bool | None
    parser_answer: str
    parser_correct: bool | None
    hard_gate: int | None
    n_tokens: int | None
    path: PathKind


@dataclass(frozen=True)
class PromptRequest:
    mode: EvalMode
    user_message: str
    thinking_prefix: str
    thinking_language_target: str
    requested_answer_language: str
    sampling: dict[str, float | int]


@dataclass(frozen=True)
class ItemRef:
    item_id: str
    source: str
    stratum: str
    split: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRecord:
    """Eval-path row with gold. Never write this to manifests/ (IDs only)."""

    item_id: str
    source: str
    task: TaskKind
    question: str
    gold: str
    prompt_language: str
    stratum: str
    options: tuple[str | None, ...] | None = None
    split: str | None = None
    extras: dict[str, str] = field(default_factory=dict)
    instruction: str | None = None
    answer_prefix: str | None = None


@dataclass(frozen=True)
class SourceSpec:
    name: str
    task: TaskKind
    role: Literal["eval", "train"]
    language: str
    present_options: Sequence[str] | None = None

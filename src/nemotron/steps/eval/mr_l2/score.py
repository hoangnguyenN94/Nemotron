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

from collections.abc import Sequence

from nemotron.steps.eval.mr_l2.closure import check_closure
from nemotron.steps.eval.mr_l2.contamination import ContaminationError
from nemotron.steps.eval.mr_l2.constants import (
    IN_LANGUAGE_THRESHOLD,
    MGSM_VI_ANSWER_PREFIX,
    TASK_BY_SOURCE,
    TOKEN_BUDGET,
)
from nemotron.steps.eval.mr_l2.hard_gate import hard_gate
from nemotron.steps.eval.mr_l2.lid import (
    LanguageDetector,
    detect_answer_language,
    score_thinking_language,
    thinking_lang_ok_for_gate,
)
from nemotron.steps.eval.mr_l2.parser import answers_match, parse_answer, parse_mgsm_prefix, split_think_answer
from nemotron.steps.eval.mr_l2.types import PathKind, ScoreResult, TaskKind


def score_trace(
    raw: str,
    *,
    question: str,
    task: TaskKind | str,
    thinking_language_target: str,
    requested_answer_language: str,
    n_tokens: int | None = None,
    gold: str | None = None,
    label: str | None = None,
    options: Sequence[str | None] | None = None,
    answer_prefix: str | None = None,
    detector: LanguageDetector | None = None,
    path: PathKind = "eval",
    in_language_threshold: float = IN_LANGUAGE_THRESHOLD,
    extract: str | None = None,
    max_tokens: int | None = None,
) -> ScoreResult:
    """
    Pure scoring of one generation.

    `gold` is eval-only. `label` is the train/RL pseudo-label. Passing gold on
    path='train' raises ContaminationError.
    """
    if path == "train" and gold is not None:
        raise ContaminationError("gold on train path")

    split = split_think_answer(raw)
    closure = check_closure(split, n_tokens=n_tokens, budget=max_tokens or TOKEN_BUDGET)
    parsed = parse_answer(
        split.answer_span,
        task,
        raw=raw if path == "eval" else "",
        options=options,
        extract=extract,
    )
    parser_answer = ""
    parser_correct: bool | None = None
    if task == "numeric":
        prefix = answer_prefix or MGSM_VI_ANSWER_PREFIX
        parser_answer = parse_mgsm_prefix(split.answer_span, prefix)
        # Eval-only: a finished generation that wrote the final "#### N" inside think.
        if path == "eval" and not parsed and not split.answer_span and not closure.truncated:
            parsed = parse_mgsm_prefix(split.thinking, prefix)
    thinking = score_thinking_language(
        split.thinking,
        thinking_language_target,
        question=question,
        detector=detector,
        threshold=in_language_threshold,
        options=options,
    )
    thinking_ok = thinking_lang_ok_for_gate(thinking)
    answer_language, answer_lang_ok = detect_answer_language(
        split.answer_span,
        requested_answer_language,
        detector=detector,
        task=task,
    )

    correct: bool | None = None
    if path == "eval" and gold is not None:
        correct = answers_match(parsed, gold, task)
        if task == "numeric":
            parser_correct = answers_match(parser_answer, gold, task)

    gate: int | None = None
    if label is not None:
        gate_parsed = parse_answer(split.answer_span, task, raw="", options=options, extract=extract)
        gate = hard_gate(
            thinking_lang_ok=thinking_ok,
            answer_lang_ok=answer_lang_ok,
            closed=closure.closed,
            answer_matches_label=answers_match(gate_parsed, label, task),
        )

    return ScoreResult(
        split=split,
        parsed_answer=parsed,
        thinking_lang_share=thinking.share,
        in_language_95=thinking.in_language_95,
        thinking_lang_ok=thinking_ok,
        answer_language=answer_language,
        answer_lang_ok=answer_lang_ok,
        closed=closure.closed,
        truncated=closure.truncated,
        loop=closure.loop,
        correct=correct,
        parser_answer=parser_answer,
        parser_correct=parser_correct,
        hard_gate=gate,
        n_tokens=n_tokens,
        path=path,
    )


def task_for_source(source: str) -> TaskKind:
    try:
        return TASK_BY_SOURCE[source]
    except KeyError as exc:
        raise ValueError(f"unknown source {source!r}") from exc

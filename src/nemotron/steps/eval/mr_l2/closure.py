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

from collections import Counter

from nemotron.steps.eval.mr_l2.constants import (
    LOOP_CONSECUTIVE,
    LOOP_COVERAGE,
    LOOP_MIN_COUNT,
    LOOP_NGRAM,
    TOKEN_BUDGET,
)
from nemotron.steps.eval.mr_l2.textnorm import tokenize_words
from nemotron.steps.eval.mr_l2.types import ClosureResult, ThinkAnswerSplit


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def is_loop(thinking: str, n: int = LOOP_NGRAM) -> bool:
    words = [w.lower() for w in tokenize_words(thinking)]
    grams = _ngrams(words, n)
    if not grams:
        return False
    counts = Counter(grams)
    top, freq = counts.most_common(1)[0]
    if freq >= LOOP_MIN_COUNT and (freq * n) / max(len(words), 1) >= LOOP_COVERAGE:
        return True
    run = 0
    prev: tuple[str, ...] | None = None
    for g in grams:
        if g == prev:
            run += 1
        else:
            run = 1
            prev = g
        if run >= LOOP_CONSECUTIVE:
            return True
    return False


def check_closure(
    split: ThinkAnswerSplit,
    n_tokens: int | None = None,
    budget: int = TOKEN_BUDGET,
) -> ClosureResult:
    close = split.think_close_present
    hit_budget = n_tokens is not None and n_tokens >= budget
    if not close:
        truncated = n_tokens is None or hit_budget
    else:
        truncated = hit_budget and not split.answer_span
    loop = is_loop(split.thinking)
    closed = close and not truncated and not loop
    return ClosureResult(
        closed=closed,
        truncated=truncated,
        loop=loop,
        think_close_present=close,
    )

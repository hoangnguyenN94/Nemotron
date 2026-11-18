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

"""Hard-gate reward for native-reasoning gated RL. Label only; never gold.

    if thinking-language fails OR answer-language fails:  -1
    elif answer matches label AND closed:                 1
    else:                                                 0

Unlabeled items are excluded by the caller, not scored 0. Thinking-language
is L2 (`in_language_95`) from eval/mr_l2. Completely empty thinking fails;
maths-only remainders are language-neutral. Empty answer spans fail; the
gate does not parse \\boxed / Đáp án from the think block.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemotron.steps.eval.mr_l2.lid import LanguageDetector, default_detector
from nemotron.steps.eval.mr_l2.score import score_trace, task_for_source
from nemotron.steps.rl.mr_gated.schema import refuse_gold


def gated_reward(
    raw: str,
    row: Mapping[str, Any],
    *,
    n_tokens: int | None = None,
    max_tokens: int | None = None,
    detector: LanguageDetector | None = None,
) -> dict[str, Any]:
    refuse_gold(row, where="gated RL reward")
    question = str(row.get("unassisted_user") or row.get("question") or "").strip()
    label = str(row.get("label") or "").strip()
    if not question or not label:
        raise ValueError("gated reward needs unassisted question and label")
    lang = str(row.get("prompt_language") or row.get("language_code") or "vi").strip() or "vi"
    source = str(row.get("source") or "")
    try:
        task = task_for_source(source) if source else "numeric"
    except ValueError:
        task = str(row.get("task") or "numeric")
    score = score_trace(
        raw,
        question=question,
        task=task,
        thinking_language_target=lang,
        requested_answer_language=lang,
        n_tokens=n_tokens,
        label=label,
        detector=detector or default_detector(),
        path="train",
        max_tokens=max_tokens,
        options=row.get("options") if isinstance(row.get("options"), (list, tuple)) else None,
        answer_prefix=row.get("answer_prefix"),
    )
    reward = int(score.hard_gate if score.hard_gate is not None else 0)
    return {
        "reward": reward,
        "parsed": score.parsed_answer,
        "closed": score.closed,
        "truncated": score.truncated,
        "loop": score.loop,
        "in_language_95": score.in_language_95,
        "thinking_lang_share": score.thinking_lang_share,
        "thinking_lang_ok": score.thinking_lang_ok,
        "answer_lang_ok": score.answer_lang_ok,
        "n_tokens": score.n_tokens,
    }

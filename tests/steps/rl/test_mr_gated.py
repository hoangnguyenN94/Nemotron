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

"""Hard-gate reward and the rl/mr_gated step runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.steps.eval.mr_l2.contamination import ContaminationError
from nemotron.steps.eval.mr_l2.hard_gate import hard_gate
from nemotron.steps.rl.mr_gated.reward import gated_reward
from nemotron.steps.rl.mr_gated.runtime import run_mr_gated
from nemotron.steps.rl.mr_gated.schema import TrainSchemaError, validate_gated_record

STEP_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "rl" / "mr_gated"
TINY_PROMPTS = STEP_DIR / "config" / "tiny_prompts.jsonl"
TINY_TRACES = STEP_DIR / "config" / "tiny_traces.jsonl"

PROMPT = {
    "item_id": "gsm8k_train_vi:tiny",
    "source": "gsm8k_train_vi",
    "split": "eligible-train",
    "task": "numeric",
    "prompt_language": "vi",
    "question": "Còn bao nhiêu?",
    "unassisted_user": "Còn bao nhiêu?",
    "label": "13",
    "path": "train",
    "arm": "approach1_gated",
}


def test_hard_gate_truth_table() -> None:
    assert hard_gate(thinking_lang_ok=True, answer_lang_ok=True, closed=True, answer_matches_label=True) == 1
    assert hard_gate(thinking_lang_ok=True, answer_lang_ok=True, closed=True, answer_matches_label=False) == 0
    assert hard_gate(thinking_lang_ok=True, answer_lang_ok=True, closed=False, answer_matches_label=True) == 0
    assert hard_gate(thinking_lang_ok=False, answer_lang_ok=True, closed=True, answer_matches_label=True) == -1
    assert hard_gate(thinking_lang_ok=True, answer_lang_ok=False, closed=True, answer_matches_label=True) == -1


def test_gated_reward_plus_one_on_vi_closed_match() -> None:
    out = gated_reward("<think>Cô ấy trừ 3 quả và còn 13.</think>\n13", PROMPT)
    assert out["reward"] == 1
    assert out["in_language_95"]
    assert out["closed"]
    assert out["answer_lang_ok"]


def test_gated_reward_zero_on_wrong_parse() -> None:
    out = gated_reward("<think>Cô ấy trừ 3 quả và còn 10.</think>\n10", PROMPT)
    assert out["reward"] == 0
    assert out["in_language_95"]


def test_gated_reward_minus_one_on_english_thinking() -> None:
    out = gated_reward(
        "<think>She subtracts 3 and there are 13 left in total.</think>\n13",
        PROMPT,
    )
    assert out["reward"] == -1
    assert not out["in_language_95"]


def test_gated_reward_refuses_gold() -> None:
    with pytest.raises(ContaminationError, match="gold"):
        gated_reward("<think>ok</think>\n13", {**PROMPT, "gold": "13"})


def test_gated_schema_refuses_hint_and_eval_source() -> None:
    validate_gated_record(PROMPT)
    with pytest.raises(TrainSchemaError, match="hint"):
        validate_gated_record({**PROMPT, "hint": "Gợi ý: trừ 3"})
    with pytest.raises(ContaminationError, match="eval source"):
        validate_gated_record({**PROMPT, "source": "mgsm_vi"})
    with pytest.raises(ContaminationError, match="eval-bench"):
        validate_gated_record(PROMPT, {"gsm8k_train_vi:tiny"})


def test_tiny_config_scores_plus_zero_minus(tmp_path: Path) -> None:
    out = tmp_path / "mr_gated"
    metrics = run_mr_gated(
        {
            "input": str(TINY_TRACES),
            "prompts": str(TINY_PROMPTS),
            "output_dir": str(out),
        }
    )
    assert metrics["n"] == 3
    assert metrics["n_plus"] == 1
    assert metrics["n_zero"] == 1
    assert metrics["n_minus"] == 1
    rows = [json.loads(line) for line in (out / "rewards.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["item_id"]: row["reward"] for row in rows}
    assert by_id == {"tiny:plus": 1, "tiny:zero": 0, "tiny:minus": -1}

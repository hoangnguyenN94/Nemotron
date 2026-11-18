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

"""L2 (in_language_95) scoring and the eval/mr_l2 step runner."""

from __future__ import annotations

import json
from pathlib import Path

from nemotron.steps.eval.mr_l2.analyze import classify_thinking
from nemotron.steps.eval.mr_l2.lid import detect_answer_language, score_thinking_language
from nemotron.steps.eval.mr_l2.runtime import run_mr_l2
from nemotron.steps.eval.mr_l2.textnorm import is_numeric_or_symbol

STEP_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "eval" / "mr_l2"
TINY_TRACES = STEP_DIR / "config" / "tiny_traces.jsonl"


def test_accented_vietnamese_passes_l2() -> None:
    score = score_thinking_language("Cô ấy còn mười ba quả trứng.", "vi")
    assert score.share == 1.0
    assert score.in_language_95


def test_unaccented_vietnamese_still_counts() -> None:
    score = score_thinking_language("Co 16 qua, khong duoc an het, con 13 qua.", "vi")
    assert score.in_language_95
    assert score.share == 1.0


def test_english_cot_fails_l2() -> None:
    score = score_thinking_language(
        "She has 16 eggs and eats 3 so there are 13 left in the basket.",
        "vi",
    )
    assert score.share == 0.0
    assert not score.in_language_95


def test_maths_only_is_allowed() -> None:
    score = score_thinking_language(r"$$16 - 3 = 13$$", "vi")
    assert score.n_scored == 0
    assert score.share == 1.0
    assert score.in_language_95


def test_option_recap_is_allowed() -> None:
    thinking = (
        "Chúng ta cần xác định 25/8 = 3.125.\n"
        "A. 3,125\n"
        "B. 31250/10000\n"
        "Vì vậy đáp án là A."
    )
    score = score_thinking_language(thinking, "vi")
    assert score.in_language_95
    assert score.share == 1.0


def test_short_english_closer_does_not_fail_long_vietnamese_think() -> None:
    thinking = ("Cô ấy còn mười ba quả trứng. " * 12) + "So answer A."
    score = score_thinking_language(thinking, "vi")
    assert score.share >= 0.95
    assert score.in_language_95


def test_english_closer_alone_still_fails() -> None:
    score = score_thinking_language("So answer A. Thus the answer is B.", "vi")
    assert not score.in_language_95
    assert score.share == 0.0


def test_numeric_and_choice_answers_are_exempt() -> None:
    lang, ok = detect_answer_language(r"\boxed{13}", "vi")
    assert lang == "numeric"
    assert ok
    assert is_numeric_or_symbol("13")
    choice, choice_ok = detect_answer_language("C", "vi", task="mcq_aj")
    assert choice == "choice"
    assert choice_ok


def test_empty_answer_is_not_exempt() -> None:
    lang, ok = detect_answer_language("", "vi", task="mcq_aj")
    assert lang == "empty"
    assert not ok


def test_analyze_labels_english_cot() -> None:
    info = classify_thinking(
        "She has 16 eggs and eats 3 so there are 13 left in the basket.",
        "Janet có 16 quả trứng.",
        target="vi",
    )
    assert info["primary"] == "english_cot"


def test_tiny_config_scores_expected_l2(tmp_path: Path) -> None:
    out = tmp_path / "mr_l2"
    metrics = run_mr_l2(
        {
            "input": str(TINY_TRACES),
            "output_dir": str(out),
            "thinking_language_target": "vi",
            "requested_answer_language": "vi",
            "threshold": 0.95,
            "task": "auto",
        }
    )
    assert metrics["n"] == 5
    rows = [json.loads(line) for line in (out / "scored.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["tiny:vi-pass"]["in_language_95"]
    assert not by_id["tiny:en-cot"]["in_language_95"]
    assert by_id["tiny:en-cot"]["l2_primary"] == "english_cot"
    assert by_id["tiny:math-only"]["in_language_95"]
    assert by_id["tiny:option-recap"]["in_language_95"]
    assert by_id["tiny:en-closer"]["in_language_95"]
    assert (out / "metrics.json").is_file()
    assert (out / "analysis.json").is_file()
    assert (out / "l2_fails.jsonl").is_file()
    fails = [json.loads(line) for line in (out / "l2_fails.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert {row["item_id"] for row in fails} == {"tiny:en-cot"}

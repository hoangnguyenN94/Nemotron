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

"""eval/mr_bench generate + W1.1 score (no live endpoint)."""

from __future__ import annotations

import json
from pathlib import Path

from nemotron.steps.eval.mr_bench.benchmarks import BENCHMARKS, default_prompt_style, spec_for
from nemotron.steps.eval.mr_bench.endpoint import Completion
from nemotron.steps.eval.mr_bench.fetch import HF_DATASETS
from nemotron.steps.eval.mr_bench.polymath import difficulty_weighted_acc, format_polymath_prompt, math_equal
from nemotron.steps.eval.mr_bench.runtime import RunConfig, run_benchmark, run_mr_bench
from nemotron.steps.eval.mr_bench.step import load_step_config
from nemotron.steps.eval.mr_bench.vmlu_csv import write_vmlu_submission_csv

STEP_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "eval" / "mr_bench"


class _FakeClient:
    def complete(self, messages, *, temperature: float, top_p: float, max_tokens: int) -> Completion:
        user = messages[0]["content"] if messages else ""
        if "trứng" in user or "16 - 3" in user or "16 − 3" in user:
            text = "<think>Cô ấy trừ ba quả trứng và còn mười ba.</think>\nĐáp án: 13"
        else:
            text = "<think>Cô ấy tính ra mười ba.</think>\n\\boxed{13}"
        return Completion(text=text, n_tokens=24, finish_reason="stop", model="dummy", raw={})


def test_tiny_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "tiny"
    metrics = run_mr_bench(
        {
            "benchmark": "tiny",
            "model_url": "http://127.0.0.1:8000/v1",
            "model_name": "dummy",
            "dry_run": True,
            "fetch": False,
            "output_dir": str(out),
            "data_root": str(tmp_path),
        },
        config_path=STEP_DIR / "config" / "tiny.yaml",
    )
    assert metrics["n"] == 2
    assert metrics["dry_run"] is True
    prompts = (out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(prompts) == 2
    first = json.loads(prompts[0])
    assert "Đáp án" in first["user_message"] or "trứng" in first["user_message"]


def test_tiny_fake_generate_scores_correct(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    metrics = run_benchmark(
        RunConfig(
            model_url="http://127.0.0.1:8000/v1",
            benchmark="tiny",
            model_name="dummy",
            data_root=tmp_path,
            out=out,
            fetch=False,
            dry_run=False,
            healthcheck=False,
        ),
        client=_FakeClient(),
    )
    assert metrics["n"] == 2
    assert metrics["n_errors"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["in_language_95_rate"] == 1.0
    rows = [json.loads(line) for line in (out / "traces.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["tiny:eggs"]["parsed_answer"] in {"13", "13.0"}
    assert by_id["tiny:eggs"]["closed"] is True
    assert by_id["tiny:eggs"]["in_language_95"] is True


def test_forced_mode_prefills_think_opener(tmp_path: Path) -> None:
    out = tmp_path / "forced"
    run_mr_bench(
        {
            "benchmark": "tiny",
            "model_url": "http://127.0.0.1:8000/v1",
            "mode": "forced",
            "thinking_language_target": "vi",
            "dry_run": True,
            "fetch": False,
            "output_dir": str(out),
            "data_root": str(tmp_path),
        },
        config_path=STEP_DIR / "config" / "tiny.yaml",
    )
    row = json.loads((out / "prompts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    roles = [m["role"] for m in row["messages"]]
    assert roles[-1] == "assistant"
    assert "<think>" in row["messages"][-1]["content"]
    assert "từng bước" in row["messages"][-1]["content"]


def test_polymath_prompt_and_dw_acc() -> None:
    prompt = format_polymath_prompt("Tính 2+2.", lang="vi")
    assert prompt.endswith(r"$\boxed{}$.")
    assert math_equal("13", "13.0")
    dw = difficulty_weighted_acc(
        {
            "low": {"scored": 10, "correct": 10},
            "medium": {"scored": 10, "correct": 5},
            "high": {"scored": 10, "correct": 0},
            "top": {"scored": 10, "correct": 0},
        }
    )
    assert dw["dw_acc"] is not None
    assert abs(dw["dw_acc"] - (1 * 1.0 + 2 * 0.5) / 15) < 1e-9


def test_default_prompt_style_and_hub_coverage() -> None:
    assert default_prompt_style("mmlu_prox_vi") == "mmlu-prox"
    assert default_prompt_style("polymath_hi") == "polymath"
    assert default_prompt_style("mgsm_vi") == "contract"
    skip = {"tiny", "mmlu_prox_vi_val", "mmlu_prox_hi_val", "mmlu_prox_en_val"}
    missing = [name for name in BENCHMARKS if name not in skip and name not in HF_DATASETS]
    assert missing == []
    spec_for("mgsm_vi")
    spec_for("polymath_hi")


def test_vmlu_submission_csv(tmp_path: Path) -> None:
    path = write_vmlu_submission_csv(
        [{"item_id": "vmlu:STEM-1", "parsed_answer": "C"}],
        tmp_path / "submission.csv",
    )
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "id,answer"
    assert "STEM-1,C" in text


def test_tiny_yaml_loads() -> None:
    cfg = load_step_config(STEP_DIR / "config" / "tiny.yaml")
    assert cfg["benchmark"] == "tiny"
    assert cfg["dry_run"] is True
    merged = load_step_config(STEP_DIR / "config" / "mgsm_vi.yaml")
    assert merged["benchmark"] == "mgsm_vi"
    assert merged["fetch"] is True
    named = {
        path.stem: load_step_config(path)
        for path in (STEP_DIR / "config").glob("*.yaml")
        if path.stem not in {"tiny", "default"}
    }
    assert named["gsm8k_hi"]["thinking_language_target"] == "hi"
    assert named["polymath_en"]["prompt_style"] == "polymath"
    assert named["mmlu_prox_hi"]["prompt_style"] == "mmlu-prox"
    for name, overlay in named.items():
        assert overlay["benchmark"] == name

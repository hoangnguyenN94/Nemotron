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

"""eval/mr_agent command plan, overlay, and metrics (no live tau2)."""

from __future__ import annotations

import json
from pathlib import Path

from nemotron.steps.eval.mr_agent.commands import plan_runs
from nemotron.steps.eval.mr_agent.metrics import aggregate_domain_summaries, summarize_results
from nemotron.steps.eval.mr_agent.overlay import UPSTREAM_GENERATE, apply_thinking_overlay
from nemotron.steps.eval.mr_agent.pins import INTAKE_ELICITATION_STRATEGY, SEATAU_SHA, TAU2_DOMAINS, TAU2_SHA
from nemotron.steps.eval.mr_agent.runtime import build_spec, config_from_mapping, run_mr_agent
from nemotron.steps.eval.mr_agent.step import load_step_config
from nemotron.steps.eval.mr_l2.constants import FORCED_OPENERS

STEP_DIR = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "eval" / "mr_agent"

FAKE_LLM_AGENT = '''from tau2.utils.llm_utils import generate

class LLMAgent:
    def _generate_next_message(self, message, state):
        if False:
            pass
''' + UPSTREAM_GENERATE + '''
class Other:
    pass
'''


def test_tiny_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "tiny"
    metrics = run_mr_agent(
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
    assert metrics["n_runs"] == 1
    assert metrics["dry_run"] is True
    plan = json.loads((out / "commands.json").read_text(encoding="utf-8"))
    assert plan["harness"] == "seatau"
    assert plan["domains"] == ["retail"]
    assert "--seatau-scenario" in plan["runs"][0]["cmd"]
    assert "l2_interaction" in plan["runs"][0]["cmd"]


def test_tau2_hi_full_domains_not_text_v1() -> None:
    spec = build_spec(
        config_from_mapping(
            {
                "benchmark": "tau2_hi",
                "model_url": "http://127.0.0.1:8000/v1",
                "model_name": "dummy",
                "mode": "forced",
            }
        )
    )
    assert spec.harness == "tau2"
    assert spec.language == "hi"
    assert spec.scenario is None
    assert spec.domains == TAU2_DOMAINS
    assert spec.num_trials == 4
    assert spec.thinking_prefix == FORCED_OPENERS["hi"]
    runs = plan_runs(spec)
    assert [r.domain for r in runs] == list(TAU2_DOMAINS)
    joined = " ".join(runs[0].cmd)
    assert "--benchmark none" in joined
    assert "--language hi" in joined
    assert "--benchmark text" not in joined
    intake = next(r for r in runs if r.domain == "intake")
    assert INTAKE_ELICITATION_STRATEGY in intake.cmd
    assert "l2_interaction" not in intake.save_to


def test_seatau_forced_prefix_in_agent_args() -> None:
    spec = build_spec(
        config_from_mapping(
            {
                "benchmark": "seatau_vi",
                "mode": "forced",
                "model_url": "http://127.0.0.1:8000/v1",
                "model_name": "nano",
            }
        )
    )
    assert spec.scenario == "l2_interaction"
    assert spec.num_trials == 3
    assert spec.thinking_prefix == FORCED_OPENERS["vi"]
    assert spec.vllm_prefill is True
    cmd = plan_runs(spec)[0].cmd
    blob = " ".join(cmd)
    assert "--seatau-scenario l2_interaction" in blob
    assert "--lang-id vi" in blob
    args = json.loads(cmd[cmd.index("--agent-llm-args") + 1])
    assert args["thinking_prefix"] == FORCED_OPENERS["vi"]
    assert args["vllm_prefill"] is True
    assert args["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_thinking_overlay_rewrites_generate(tmp_path: Path) -> None:
    path = tmp_path / "llm_agent.py"
    path.write_text(FAKE_LLM_AGENT, encoding="utf-8")
    apply_thinking_overlay(path, thinking_src=STEP_DIR / "overlay" / "thinking.py")
    text = path.read_text(encoding="utf-8")
    assert "prepare_thinking_generate" in text
    assert text.count("prepare_thinking_generate") == 2  # import + call
    assert "continue_final_message" in (tmp_path / "thinking_overlay.py").read_text(encoding="utf-8")
    apply_thinking_overlay(path, thinking_src=STEP_DIR / "overlay" / "thinking.py")
    assert path.read_text(encoding="utf-8") == text


def test_metrics_pass1_and_overlap(tmp_path: Path) -> None:
    payload = {
        "info": {"environment_info": {"domain_name": "retail"}, "lang_id": "vi"},
        "simulations": [
            {"task_id": "1", "termination_reason": "done", "reward_info": {"reward": 1.0, "language_correctness": {"score": 1.0}}},
            {"task_id": "2", "termination_reason": "done", "reward_info": {"reward": 0.0, "language_correctness": {"score": 0.5}}},
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = summarize_results(path)
    assert row["n_tasks"] == 2
    assert row["pass1"] == 0.5
    assert row["language_correctness"] == 0.75
    airline = dict(row)
    airline["domain"] = "airline"
    airline["n_success"] = 1
    airline["n_reward"] = 1
    bank = {"domain": "banking_knowledge", "n_success": 0, "n_reward": 10, "language_correctness": None}
    metrics = aggregate_domain_summaries([airline, row, bank], suite="tau2_hi", mode="free")
    assert metrics["n_success"] == 2
    assert metrics["n_reward"] == 13
    assert metrics["cs_overlap_pass1"] == 2 / 3


def test_yaml_and_pins() -> None:
    tiny = load_step_config(STEP_DIR / "config" / "tiny.yaml")
    assert tiny["benchmark"] == "tiny"
    assert tiny["dry_run"] is True
    hi = load_step_config(STEP_DIR / "config" / "tau2_hi.yaml")
    assert hi["benchmark"] == "tau2_hi"
    assert hi["num_trials"] == 4
    assert hi["fetch"] is True
    spec = build_spec(config_from_mapping(hi, config_path=STEP_DIR / "config" / "tau2_hi.yaml"))
    assert spec.scenario is None
    assert spec.domains == TAU2_DOMAINS
    assert len(SEATAU_SHA) == 40
    assert len(TAU2_SHA) == 40

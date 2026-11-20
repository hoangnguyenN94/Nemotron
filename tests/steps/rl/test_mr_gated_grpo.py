# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Training data, plugin hook and config that run GRPO with the rl/mr_gated reward."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemotron.steps._runners import nemo_rl
from nemotron.steps.rl.mr_gated import data
from nemotron.steps.rl.mr_gated.reward import gated_reward
from nemotron.steps.rl.mr_gated.schema import validate_gated_record

CONFIG = Path(__file__).resolve().parents[3] / "src/nemotron/steps/rl/nemo_rl/rlvr/config/nano4b_mr_gated.yaml"


def test_scope_key_counts_steps_and_operations() -> None:
    solution = "She has 16 - 3 = <<16-3=13>>13 eggs.\nShe makes 13 * 2 = <<13*2=26>>26 dollars.\n#### 26"

    assert data.scope_key(solution) == (2, "*-")
    assert data.scope_key("No annotated calculation.\n#### 5") == (0, "")


def test_spread_subset_keeps_each_kind_in_proportion() -> None:
    items = [("easy", i) for i in range(80)] + [("hard", i) for i in range(20)]

    subset = data.spread_subset(items, 0.25, key=lambda item: item)

    assert len(subset) == len(set(subset)) == 25
    assert Counter(kind for kind, _ in subset) == {"easy": 20, "hard": 5}
    # Picks cover the whole range of a kind, not only its start.
    assert max(i for kind, i in subset if kind == "easy") > 70


def test_gated_rows_satisfy_the_prompt_contract_and_score() -> None:
    row = data.gated_row(data.EN_SOURCE, 7, "train", "en", " How many eggs are left? ", " 1,300 ")

    validate_gated_record(row)
    assert row["item_id"] == "gsm8k_train_en:7"
    assert row["unassisted_user"] == row["question"] == "How many eggs are left?"
    assert row["label"] == "1300"
    # The English source is not in the L2 source table, so the row's own task is used.
    assert gated_reward("There are 1300 eggs left in the basket.\n</think>\n1300", row)["parsed"] == "1300"


def test_plugins_register_before_the_upstream_example_runs(tmp_path, monkeypatch) -> None:
    (tmp_path / "fake_plugin.py").write_text(
        "import sys\ndef register():\n    sys.modules[__name__].registered = True\n", encoding="utf-8"
    )
    script = tmp_path / "run_grpo.py"
    script.write_text(
        "import json, sys, fake_plugin\n"
        "json.dump({'registered': fake_plugin.registered, 'argv': sys.argv[1:]}, open(sys.argv[2] + '.seen', 'w'))\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("env:\n  plugins: [fake_plugin]\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["step.py", "--config", str(config), "grpo.max_num_steps=3"])
    monkeypatch.setattr(nemo_rl.os, "execvp", lambda *args: pytest.fail("plugins must run in-process"))

    nemo_rl.exec_or_run_nemo_rl_grpo(default_config=config, upstream_script=str(script), description="test")

    seen = OmegaConf.load(str(config) + ".seen")
    assert seen.registered is True
    assert list(seen.argv) == ["--config", str(config), "grpo.max_num_steps=3"]


def test_configs_without_plugins_still_exec_the_upstream_example(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("env:\n  should_use_nemo_gym: false\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(sys, "argv", ["step.py", "--config", str(config)])
    monkeypatch.setattr(nemo_rl.os, "execvp", lambda *args: calls.append(args))

    nemo_rl.exec_or_run_nemo_rl_grpo(default_config=config, upstream_script="run_grpo.py", description="test")

    assert calls == [(sys.executable, [sys.executable, "run_grpo.py", "--config", str(config)])]


def test_mr_gated_grpo_config(monkeypatch) -> None:
    monkeypatch.setenv("RL_OUTPUT_DIR", "/out")
    monkeypatch.setenv("RL_POLICY_MODEL", "/models/sft")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    cfg = OmegaConf.to_container(nemo_rl.load_nemo_rl_step_config(CONFIG), resolve=True)

    assert nemo_rl.nemo_rl_plugins(CONFIG) == ["nemotron.steps.rl.mr_gated.nemo_rl_plugin"]
    # The reward imports eval/mr_l2, which is outside the step's own category.
    assert cfg["run"]["env"]["source_branches"] == ["eval"]
    assert cfg["policy"]["tokenizer"]["chat_template_kwargs"] == {"enable_thinking": True}
    for split, name in (("train", "train"), ("validation", "val")):
        assert cfg["data"][split]["dataset_name"] == "mr_gated_gsm8k"
        assert cfg["data"][split]["split"] == name
        assert cfg["data"][split]["english_fraction"] == 0.25
    assert cfg["data"]["default"]["env_name"] == "mr_gated" in cfg["env"]
    # The gated prompt contract forbids anything but the plain question.
    assert cfg["data"]["default"]["prompt_file"] is None
    generation = cfg["policy"]["generation"]
    assert cfg["env"]["mr_gated"] == {
        "tokenizer_name": "/models/sft",
        "max_tokens": generation["max_new_tokens"],
        "log_every": 448,
    }
    assert generation["top_p"] == 0.95
    assert generation["max_new_tokens"] < cfg["policy"]["max_total_sequence_length"]
    # One epoch: 6,725 Vietnamese + 1,681 English prompts at 56 per step.
    grpo = cfg["grpo"]
    assert grpo["max_num_steps"] == (6725 + 6725 // 4) // grpo["num_prompts_per_step"] == 150
    # Rollout, training and validation batches must split evenly over the 8 GPUs of one node.
    shards = cfg["cluster"]["num_nodes"] * cfg["cluster"]["gpus_per_node"]
    for size in (
        grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"],
        cfg["policy"]["train_global_batch_size"],
        grpo["max_val_samples"],
        grpo["val_batch_size"],
    ):
        assert size % shards == 0

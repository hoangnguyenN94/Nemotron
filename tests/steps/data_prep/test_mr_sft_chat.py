# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks and record building for ``steps/data_prep/mr_sft_chat``."""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from nemotron.steps.data_prep.mr_sft_chat import runtime

from .._step_helpers import assert_step_static, step_dir

ROW = {
    "id": "vi-math-000001",
    "language_code": "vi",
    "domain": "math",
    "question": " Tính 2 + 3. ",
    "thinking": "Ta cộng 2 và 3 được 5.",
    "output": "Đáp án là 5.",
    "original_question": "Compute 2 + 3.",
    "original_thinking": "We add 2 and 3 to get 5.",
    "original_output": "The answer is 5.",
}


def test_mr_sft_chat_static() -> None:
    assert_step_static(
        step_dir(__file__, "data_prep", "mr_sft_chat"),
        expected_name="steps/data_prep/mr_sft_chat",
        expected_launch="python",
        expected_default_config="default",
    )


def test_row_gives_target_language_and_english_examples() -> None:
    records, stats = runtime.build_records([ROW], language="vi")

    assert [record["id"] for record in records] == ["vi-math-000001:vi", "vi-math-000001:en"]
    assert stats == {"vi_kept": 1, "en_kept": 1}
    vietnamese, english = (record["messages"] for record in records)
    assert vietnamese == [
        {"role": "user", "content": "Tính 2 + 3."},
        {"role": "assistant", "reasoning_content": "Ta cộng 2 và 3 được 5.", "content": "Đáp án là 5."},
    ]
    assert english == [
        {"role": "user", "content": "Compute 2 + 3."},
        {"role": "assistant", "reasoning_content": "We add 2 and 3 to get 5.", "content": "The answer is 5."},
    ]


def test_english_can_be_left_out() -> None:
    records, stats = runtime.build_records([ROW], language="vi", include_english=False)

    assert [record["language"] for record in records] == ["vi"]
    assert stats == {"vi_kept": 1}


def test_empty_and_too_long_examples_are_dropped_per_language() -> None:
    rows = [
        ROW,
        {**ROW, "id": "vi-math-000002", "thinking": "  ", "original_output": None},
        {**ROW, "id": "vi-math-000003", "thinking": "dài " * 50},
    ]
    count_tokens = lambda messages: len(messages[1]["reasoning_content"].split())  # noqa: E731

    records, stats = runtime.build_records(rows, language="vi", count_tokens=count_tokens, max_tokens=20)

    assert [record["id"] for record in records] == [
        "vi-math-000001:vi",
        "vi-math-000001:en",
        "vi-math-000003:en",
    ]
    assert stats == {"vi_kept": 1, "en_kept": 2, "vi_empty": 1, "en_empty": 1, "vi_too_long": 1}


def test_run_writes_jsonl_blends_and_stats(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeSplit(list):
        def select(self, indices):
            return FakeSplit(self[i] for i in indices)

    def load_dataset(name, subset, split):
        calls.append((name, subset, split))
        return FakeSplit({**ROW, "id": f"vi-math-{split}-{i}"} for i in range(3))

    datasets = ModuleType("datasets")
    datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    cfg = {
        "dataset": "org/reasoning",
        "subset": "vi",
        "include_english": True,
        "splits": {"train": "train", "valid": "test"},
        "max_tokens": None,
        "max_rows": 2,
        "output_dir": str(tmp_path),
    }

    stats = runtime.run_mr_sft_chat(cfg)

    assert calls == [("org/reasoning", "vi", "train"), ("org/reasoning", "vi", "test")]
    assert stats["splits"]["valid"] == {"source_split": "test", "rows": 2, "examples": 4, "vi_kept": 2, "en_kept": 2}
    lines = (tmp_path / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [
        "vi-math-test-0:vi",
        "vi-math-test-0:en",
        "vi-math-test-1:vi",
        "vi-math-test-1:en",
    ]
    assert "Đáp án" in lines[0]  # written as UTF-8, not escaped
    blend = json.loads((tmp_path / "blend_train.json").read_text())
    assert blend == {"datasets": [{"name": "mr_sft_vi_train", "path": str(tmp_path / "train.jsonl")}]}
    assert json.loads((tmp_path / "stats.json").read_text()) == stats


def test_default_config_uses_only_the_vietnamese_subset_and_fits_8k_packing() -> None:
    step = step_dir(__file__, "data_prep", "mr_sft_chat")
    cfg = OmegaConf.load(step / "config" / "default.yaml")
    packing = OmegaConf.load(step.parent / "sft_packing" / "config" / "nano4b_mr_vi.yaml")

    assert cfg.subset == "vi"
    assert cfg.include_english is True
    assert dict(cfg.splits) == {"train": "train", "valid": "test"}
    assert cfg.tokenizer == packing.tokenizer.model
    assert cfg.max_tokens < packing.pack_size

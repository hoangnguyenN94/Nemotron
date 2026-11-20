# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks and PEFT import fallbacks for ``steps/peft/megatron_bridge``."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nemotron.steps._runners import megatron_bridge as runner

from .._step_helpers import assert_step_static, step_dir


def test_peft_megatron_bridge_static() -> None:
    assert_step_static(
        step_dir(__file__, "peft", "megatron_bridge"),
        expected_name="steps/peft/megatron_bridge",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_lightning35_overlay_inherits_default_yaml() -> None:
    from nemo_runspec.config.loader import load_config

    step = step_dir(__file__, "peft", "megatron_bridge")
    cfg = load_config(step / "config" / "lightning35.yaml")
    assert "defaults" not in cfg
    assert cfg.recipe._target_ == (
        "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning_peft_config"
    )
    assert cfg.recipe.packed_sequence is None
    assert cfg.recipe.peft is None
    assert cfg.peft.type == "lora"
    assert cfg.peft.dim == 32
    assert "in_proj" in list(cfg.peft.target_modules)
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.train.global_batch_size == 128
    assert cfg.hf_load.inherit_save_as_load is False
    assert cfg.checkpoint.fully_parallel_save is False
    assert cfg.run.env.container_image == "nvcr.io/nvidia/nemo:26.08"


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def _peft_yaml(**overrides: object) -> dict:
    block = {
        "type": "lora",
        "dim": 32,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["linear_qkv", "in_proj"],
    }
    block.update(overrides)
    return {"peft": block}


def test_maybe_apply_peft_skips_when_block_missing() -> None:
    cfg = SimpleNamespace(peft=None)
    runner._maybe_apply_peft(cfg, {})
    assert cfg.peft is None


def test_maybe_apply_peft_uses_dataset_utils_on_nemo_26_08(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nemo:26.08 / Megatron-Bridge 0c565c9 moved default_peft_config here."""
    calls: list[tuple[object, dict]] = []

    def fake_helper(*, peft_scheme, **kwargs):
        calls.append((peft_scheme, kwargs))
        return SimpleNamespace(scheme=peft_scheme, **kwargs)

    dataset_utils = ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    dataset_utils.default_peft_config = fake_helper
    _install_module(monkeypatch, "megatron.bridge.recipes.utils.dataset_utils", dataset_utils)

    import importlib

    original_import = importlib.import_module

    def boom(name: str, *args, **kwargs):
        if name.endswith("finetune_utils"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", boom)

    cfg = SimpleNamespace(peft=None)
    runner._maybe_apply_peft(cfg, _peft_yaml())

    assert calls == [
        (
            "lora",
            {
                "dim": 32,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": ["linear_qkv", "in_proj"],
            },
        )
    ]
    assert cfg.peft.scheme == "lora"
    assert cfg.peft.dim == 32


def test_maybe_apply_peft_constructs_lora_when_helpers_are_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLoRA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    lora_mod = ModuleType("megatron.bridge.peft.lora")
    lora_mod.LoRA = FakeLoRA
    _install_module(monkeypatch, "megatron.bridge.peft.lora", lora_mod)

    import importlib

    original_import = importlib.import_module

    def boom(name: str, *args, **kwargs):
        if name.endswith(("finetune_utils", "dataset_utils")):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", boom)

    cfg = SimpleNamespace(peft=None)
    runner._maybe_apply_peft(cfg, _peft_yaml(type="lora", dim=16))

    assert isinstance(cfg.peft, FakeLoRA)
    assert cfg.peft.kwargs["dim"] == 16
    assert cfg.peft.kwargs["target_modules"] == ["linear_qkv", "in_proj"]


def _peft_cfg(pretrained: str | None = None) -> SimpleNamespace:
    checkpoint = SimpleNamespace(pretrained_checkpoint=pretrained, load=None, save="/ckpt/adapter")
    return SimpleNamespace(checkpoint=checkpoint, model="recipe-model")


def _install_checkpoint_utils(
    monkeypatch: pytest.MonkeyPatch, *, megatron_dirs: set[str], hf_dirs: set[str] | None
) -> None:
    """Stub Megatron-Bridge checkpoint detection; ``hf_dirs=None`` models a release without HF bases."""
    module = ModuleType("megatron.bridge.training.utils.checkpoint_utils")
    module.checkpoint_exists = lambda path: path in megatron_dirs
    if hf_dirs is not None:
        module.is_hf_checkpoint_dir = lambda path: path in hf_dirs
    _install_module(monkeypatch, "megatron.bridge.training.utils.checkpoint_utils", module)


def _peft_container(**extra: object) -> dict:
    return {**_peft_yaml(), "hf_load": {"inherit_save_as_load": False}, **extra}


def test_peft_base_resolves_hf_model_id_to_its_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    _install_checkpoint_utils(monkeypatch, megatron_dirs=set(), hf_dirs={"/hf/snapshots/abc"})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: "/hf/snapshots/abc")

    cfg = _peft_cfg()
    runner._maybe_load_hf_weights(cfg, _peft_container(hf_model_path="org/model", trust_remote_code=True))

    assert cfg.checkpoint.pretrained_checkpoint == "/hf/snapshots/abc"
    assert cfg.checkpoint.hf_trust_remote_code is True
    # Megatron-Bridge loads the base in its PEFT hook; the runner must not build it too.
    assert cfg.model == "recipe-model"
    assert cfg.checkpoint.load is None


def test_peft_base_keeps_explicit_hf_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_checkpoint_utils(monkeypatch, megatron_dirs=set(), hf_dirs={"/models/base-hf"})

    cfg = _peft_cfg("/models/base-hf")
    runner._maybe_load_hf_weights(cfg, _peft_container(hf_model_path="org/other-model"))

    assert cfg.checkpoint.pretrained_checkpoint == "/models/base-hf"
    assert cfg.checkpoint.hf_trust_remote_code is False


def test_peft_base_keeps_megatron_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_checkpoint_utils(monkeypatch, megatron_dirs={"/ckpt/base"}, hf_dirs=set())

    cfg = _peft_cfg("/ckpt/base")
    runner._maybe_load_hf_weights(cfg, _peft_container())

    assert cfg.checkpoint.pretrained_checkpoint == "/ckpt/base"
    assert not hasattr(cfg.checkpoint, "hf_trust_remote_code")


@pytest.mark.parametrize("hf_dirs", [None, set()], ids=["old-megatron-bridge", "not-an-hf-directory"])
def test_peft_base_without_usable_checkpoint_fails(monkeypatch: pytest.MonkeyPatch, hf_dirs) -> None:
    import huggingface_hub

    _install_checkpoint_utils(monkeypatch, megatron_dirs=set(), hf_dirs=hf_dirs)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: "/hf/snapshots/abc")

    with pytest.raises(ValueError, match="PEFT requires checkpoint.pretrained_checkpoint"):
        runner._maybe_load_hf_weights(_peft_cfg(), _peft_container(hf_model_path="org/model"))


def test_nano4b_mr_vi_config_is_standalone_lora_at_8k() -> None:
    from nemo_runspec.config.loader import load_config

    step = step_dir(__file__, "peft", "megatron_bridge")
    cfg = load_config(step / "config" / "nano4b_mr_vi.yaml")
    packing = load_config(step.parents[1] / "data_prep" / "sft_packing" / "config" / "nano4b_mr_vi.yaml")

    # The 4B recipe takes no arguments: nothing but the target may reach it.
    assert dict(cfg.recipe) == {
        "_target_": "megatron.bridge.recipes.nemotronh.nemotron_3_nano_4b.nemotron_3_nano_4b_peft_config"
    }
    assert cfg.hf_model_path == packing.tokenizer.model == "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert cfg.peft.type == "lora"
    assert (cfg.peft.dim, cfg.peft.alpha) == (16, 32)
    assert {"linear_qkv", "linear_fc1", "in_proj", "out_proj"} <= set(cfg.peft.target_modules)
    assert cfg.model.seq_length == cfg.dataset.seq_length == packing.pack_size == 8192
    assert cfg.dataset.packed_sequence_specs.packed_sequence_size == packing.pack_size
    # One 8-GPU node with TP=PP=CP=1: the global batch must divide evenly across ranks.
    assert cfg.train.global_batch_size % 8 == 0
    assert cfg.scheduler.lr_decay_iters == cfg.train.train_iters
    assert cfg.hf_load.inherit_save_as_load is False
    assert cfg.checkpoint.also_save_hf_checkpoint is True
    assert cfg.run.env.container_image == "nvcr.io/nvidia/nemo:26.08"

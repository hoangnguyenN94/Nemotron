#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/data_prep/mr_sft_chat"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

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

"""Build chat-format SFT JSONL from one language subset of Tiny Aya multilingual reasoning."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import OmegaConf

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)
from nemotron.steps.data_prep.mr_sft_chat.runtime import run_mr_sft_chat

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    cfg = OmegaConf.to_container(
        apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides),
        resolve=True,
    )
    run_mr_sft_chat(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/eval/mr_agent"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
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

"""Native-reasoning agent benches (SEATau-vi, τ-bench 2.0 HI/EN)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.eval.mr_agent.runtime import run_mr_agent

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def load_step_config(path: Path) -> dict[str, Any]:
    """Load YAML and merge a local `defaults: other.yaml` chain."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must be a mapping")
    defaults = raw.pop("defaults", None)
    if not defaults:
        return raw
    items = defaults if isinstance(defaults, list) else [defaults]
    merged: dict[str, Any] = {}
    for item in items:
        default_path = Path(str(item))
        if not default_path.is_absolute():
            default_path = path.parent / default_path
        merged.update(load_step_config(default_path))
    merged.update(raw)
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG))
    args, _ = parser.parse_known_args()
    config_path = Path(args.config)
    cfg = load_step_config(config_path)
    run_mr_agent(cfg, config_path=config_path)


if __name__ == "__main__":
    main()

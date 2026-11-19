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

"""Fetch a pinned SEATau / tau2-bench checkout and apply Native-reasoning overlays."""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

from nemotron.steps.eval.mr_agent.overlay import apply_harness_overlay
from nemotron.steps.eval.mr_agent.pins import LID_URL, SEATAU_GIT, SEATAU_SHA, TAU2_GIT, TAU2_SHA

LOGGER = logging.getLogger(__name__)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    LOGGER.info(" ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def clone_at(url: str, sha: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    git_dir = dest / ".git"
    if git_dir.is_dir():
        try:
            current = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=dest, text=True
            ).strip()
        except subprocess.CalledProcessError:
            current = ""
        if current.startswith(sha):
            LOGGER.info("reuse %s at %s", dest, sha)
            return
        LOGGER.info("refresh %s (have %s, want %s)", dest, current, sha)
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q", str(dest)])
    _run(["git", "remote", "add", "origin", url], cwd=dest)
    _run(["git", "fetch", "-q", "--depth", "1", "origin", sha], cwd=dest)
    _run(["git", "checkout", "-q", "-f", sha], cwd=dest)


def ensure_lid(clone_root: Path) -> Path:
    dest = clone_root / "data" / "models" / "lid.176.bin"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 100_000:
        return dest
    LOGGER.info("download FastText LID -> %s", dest)
    with urllib.request.urlopen(LID_URL, timeout=120) as resp, dest.open("wb") as handle:
        shutil.copyfileobj(resp, handle)
    return dest


def prepare_harness(harness: str, dest: Path, *, overlay: bool = True, lid: bool = False) -> Path:
    if harness == "seatau":
        clone_at(SEATAU_GIT, SEATAU_SHA, dest)
    elif harness == "tau2":
        clone_at(TAU2_GIT, TAU2_SHA, dest)
    else:
        raise ValueError(f"unknown harness {harness!r}")
    if overlay:
        apply_harness_overlay(dest, harness=harness)
    if lid:
        ensure_lid(dest)
    return dest

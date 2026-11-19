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

"""Summarize tau2 / SEATau `results.json` into W1.1 headline numbers."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemotron.steps.eval.mr_agent.pins import CS_OVERLAP_DOMAINS


def _reward(sim: Mapping[str, Any]) -> float | None:
    info = sim.get("reward_info") or {}
    if not isinstance(info, dict):
        return None
    raw = info.get("reward")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _language_score(sim: Mapping[str, Any]) -> float | None:
    info = sim.get("reward_info") or {}
    if not isinstance(info, dict):
        return None
    lang = info.get("language_correctness")
    if not isinstance(lang, dict):
        return None
    raw = lang.get("score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def summarize_results(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": str(path), "size": path.stat().st_size if path.is_file() else 0}
    if not path.is_file():
        rec["error"] = "missing"
        return rec
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("info") or {}
    env = info.get("environment_info") or {}
    sims = data.get("simulations") or []
    rewards: list[float] = []
    lang_scores: list[float] = []
    terms: Counter[str] = Counter()
    success = 0
    none_reward = 0
    task_ids: set[str] = set()
    for sim in sims:
        if not isinstance(sim, dict):
            continue
        task_ids.add(str(sim.get("task_id")))
        terms[str(sim.get("termination_reason") or "unknown")] += 1
        reward = _reward(sim)
        if reward is None:
            none_reward += 1
        else:
            rewards.append(reward)
            if abs(reward - 1.0) <= 1e-6:
                success += 1
        lang = _language_score(sim)
        if lang is not None:
            lang_scores.append(lang)
    rec.update(
        {
            "domain": env.get("domain_name") if isinstance(env, dict) else None,
            "lang_id": info.get("lang_id") or info.get("language"),
            "n_sims": len(sims),
            "n_tasks": len(task_ids),
            "n_reward": len(rewards),
            "n_none_reward": none_reward,
            "n_success": success,
            "avg_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "pass1": (success / len(rewards)) if rewards else None,
            "language_correctness": (sum(lang_scores) / len(lang_scores)) if lang_scores else None,
            "termination": dict(terms),
            "timestamp": data.get("timestamp"),
        }
    )
    return rec


def aggregate_domain_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    mode: str,
) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    n_success = n_reward = 0
    overlap_success = overlap_reward = 0
    lang_mass: list[float] = []
    for row in rows:
        domain = str(row.get("domain") or "unknown")
        by_domain[domain] = dict(row)
        n_s = int(row.get("n_success") or 0)
        n_r = int(row.get("n_reward") or 0)
        n_success += n_s
        n_reward += n_r
        if domain in CS_OVERLAP_DOMAINS:
            overlap_success += n_s
            overlap_reward += n_r
        lang = row.get("language_correctness")
        if isinstance(lang, (int, float)):
            lang_mass.append(float(lang))
    metrics: dict[str, Any] = {
        "suite": suite,
        "mode": mode,
        "n_domains": len(by_domain),
        "n_reward": n_reward,
        "n_success": n_success,
        "pass1": (n_success / n_reward) if n_reward else None,
        "cs_overlap_pass1": (overlap_success / overlap_reward) if overlap_reward else None,
        "language_correctness": (sum(lang_mass) / len(lang_mass)) if lang_mass else None,
        "by_domain": by_domain,
        "w11_scores": True,
    }
    return metrics

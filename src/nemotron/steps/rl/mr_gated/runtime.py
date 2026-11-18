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

"""Score rollouts with the hard-gate reward and write a validation report."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nemotron.steps.eval.mr_l2.runtime import _raw_output, load_traces, resolve_path
from nemotron.steps.rl.mr_gated.reward import gated_reward
from nemotron.steps.rl.mr_gated.schema import GATED_ARM, validate_gated_record

LOGGER = logging.getLogger(__name__)
STEP_DIR = Path(__file__).resolve().parent


def run_mr_gated(cfg: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    base = config_path.parent if config_path is not None else Path.cwd()
    traces_path = _resolve_existing(cfg.get("input") or cfg.get("traces"), base=base, kind="input")
    prompts_raw = cfg.get("prompts")
    prompts_path = _resolve_existing(prompts_raw, base=base, kind="prompts") if prompts_raw else None
    output_dir = resolve_path(cfg.get("output_dir") or "./output/mr_gated", base=base)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = int(cfg.get("max_samples") or 0)
    eval_ids = _load_eval_ids(cfg.get("eval_ids"), base=base)

    traces = load_traces(traces_path, max_samples=max_samples)
    prompts = _load_prompts(prompts_path, eval_ids=eval_ids) if prompts_path else {}
    if cfg.get("dry_run"):
        summary = {
            "n_traces": len(traces),
            "n_prompts": len(prompts),
            "dry_run": True,
            "input": str(traces_path),
            "prompts": str(prompts_path) if prompts_path else None,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.dry-run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("dry-run: %s traces, %s prompts", len(traces), len(prompts))
        return summary

    scored: list[dict[str, Any]] = []
    for rec in traces:
        row = _join_prompt(rec, prompts, eval_ids=eval_ids)
        raw = _raw_output(rec)
        n_tokens = rec.get("n_tokens")
        try:
            n_tokens_i = int(n_tokens) if n_tokens is not None else None
        except (TypeError, ValueError):
            n_tokens_i = None
        reward = gated_reward(raw, row, n_tokens=n_tokens_i, max_tokens=rec.get("max_tokens"))
        scored.append(
            {
                "item_id": row.get("item_id"),
                "source": row.get("source"),
                "task": row.get("task"),
                "label": row.get("label"),
                **reward,
            }
        )

    metrics = summarize_rewards(scored)
    metrics["input"] = str(traces_path)
    metrics["prompts"] = str(prompts_path) if prompts_path else None
    metrics["output_dir"] = str(output_dir)
    metrics["n_prompts"] = len(prompts)

    with (output_dir / "rewards.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "hard-gate n=%s +1=%s 0=%s -1=%s mean=%.3f → %s",
        metrics["n"],
        metrics["n_plus"],
        metrics["n_zero"],
        metrics["n_minus"],
        metrics["reward_mean"] or 0.0,
        output_dir,
    )
    return metrics


def summarize_rewards(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts: Counter[int] = Counter(int(row["reward"]) for row in rows)
    n_plus = counts.get(1, 0)
    n_zero = counts.get(0, 0)
    n_minus = counts.get(-1, 0)
    rewards = [int(row["reward"]) for row in rows]
    n_l2 = sum(1 for row in rows if row.get("in_language_95"))
    n_closed = sum(1 for row in rows if row.get("closed"))
    return {
        "n": n,
        "n_plus": n_plus,
        "n_zero": n_zero,
        "n_minus": n_minus,
        "reward_mean": (sum(rewards) / n if n else None),
        "in_language_95_rate": (n_l2 / n if n else None),
        "closed_rate": (n_closed / n if n else None),
        "formula": "if language fails: -1; elif parsed==label and closed: +1; else 0",
    }


def _join_prompt(
    rec: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Any]],
    *,
    eval_ids: Iterable[str],
) -> dict[str, Any]:
    item_id = str(rec.get("item_id") or "")
    if prompts:
        if item_id not in prompts:
            raise KeyError(f"trace {item_id!r} has no gated prompt")
        return dict(prompts[item_id])
    row = {
        key: value
        for key, value in rec.items()
        if key not in {"raw_output", "raw", "generation", "thinking", "answer_span"}
    }
    row.setdefault("path", "train")
    row.setdefault("arm", GATED_ARM)
    if "unassisted_user" not in row and row.get("question"):
        row["unassisted_user"] = str(row.get("question") or "").rstrip()
    validate_gated_record(row, eval_ids)
    return row


def _load_prompts(path: Path, *, eval_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            validate_gated_record(rec, eval_ids)
            item_id = str(rec.get("item_id") or "")
            if not item_id:
                raise ValueError(f"gated prompt missing item_id in {path}")
            rows[item_id] = rec
    return rows


def _load_eval_ids(raw: Any, *, base: Path) -> set[str]:
    if not raw:
        return set()
    path = _resolve_existing(raw, base=base, kind="eval_ids")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _resolve_existing(raw: Any, *, base: Path, kind: str) -> Path:
    if not raw:
        raise ValueError(f"config.{kind} is required")
    path = resolve_path(raw, base=base)
    if path.exists():
        return path
    bundled = STEP_DIR / "config" / Path(str(raw)).name
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"{kind} not found: {raw}")

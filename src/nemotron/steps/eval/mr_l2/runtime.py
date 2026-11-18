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

"""Score generation traces for native-reasoning L2 and write an analysis report."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemotron.steps.eval.mr_l2.analyze import classify_record, summarize_l2
from nemotron.steps.eval.mr_l2.constants import IN_LANGUAGE_THRESHOLD, THINK_CLOSE, THINK_OPEN, bench_language
from nemotron.steps.eval.mr_l2.score import score_trace, task_for_source
from nemotron.steps.eval.mr_l2.types import TaskKind

LOGGER = logging.getLogger(__name__)
STEP_DIR = Path(__file__).resolve().parent


def run_mr_l2(cfg: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    """Score traces.jsonl, write scored traces + L2 analysis under output_dir."""
    base = config_path.parent if config_path is not None else Path.cwd()
    input_path = resolve_input(cfg.get("input"), base=base)
    output_dir = resolve_path(cfg.get("output_dir") or "./output/mr_l2", base=base)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(cfg.get("max_samples") or 0)
    traces = load_traces(input_path, max_samples=max_samples)
    if cfg.get("dry_run"):
        summary = {
            "n": len(traces),
            "dry_run": True,
            "input": str(input_path),
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.dry-run.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("dry-run: %s traces from %s", len(traces), input_path)
        return summary

    default_target = str(cfg.get("thinking_language_target") or "vi")
    default_answer = str(cfg.get("requested_answer_language") or default_target)
    threshold = float(cfg.get("threshold") or IN_LANGUAGE_THRESHOLD)
    default_task = cfg.get("task") or "auto"

    scored: list[dict[str, Any]] = []
    for rec in traces:
        row = score_record(
            rec,
            default_target=default_target,
            default_answer=default_answer,
            default_task=default_task,
            threshold=threshold,
        )
        scored.append(row)

    metrics = summarize_l2(scored)
    metrics["input"] = str(input_path)
    metrics["output_dir"] = str(output_dir)
    metrics["threshold"] = threshold
    metrics["thinking_language_target"] = default_target

    scored_path = output_dir / "scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fail_path = output_dir / "l2_fails.jsonl"
    with fail_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            if row.get("in_language_95"):
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "formula": metrics["formula"],
                "in_language_95_rate": metrics["in_language_95_rate"],
                "n": metrics["n"],
                "n_in_language_95": metrics["n_in_language_95"],
                "share_buckets": metrics["share_buckets"],
                "fail_primary": metrics["fail_primary"],
                "by_source": metrics["by_source"],
                "fail_samples": metrics["fail_samples"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "L2 %s/%s (%.1f%%) → %s",
        metrics["n_in_language_95"],
        metrics["n"],
        100.0 * (metrics["in_language_95_rate"] or 0.0),
        output_dir,
    )
    return metrics


def score_record(
    rec: Mapping[str, Any],
    *,
    default_target: str,
    default_answer: str,
    default_task: str,
    threshold: float,
) -> dict[str, Any]:
    source = str(rec.get("source") or "")
    task = _resolve_task(rec.get("task") or default_task, source)
    target = str(
        rec.get("thinking_language_target")
        or rec.get("prompt_language")
        or (bench_language(source) if source else default_target)
        or default_target
    )
    answer_lang = str(rec.get("requested_answer_language") or target or default_answer)
    question = str(rec.get("question") or rec.get("user_message") or "")
    options = rec.get("options")
    if options is not None and not isinstance(options, (list, tuple)):
        options = None
    raw = _raw_output(rec)
    n_tokens = rec.get("n_tokens")
    try:
        n_tokens_i = int(n_tokens) if n_tokens is not None else None
    except (TypeError, ValueError):
        n_tokens_i = None
    gold = rec.get("gold")
    gold_s = None if gold is None else str(gold)
    score = score_trace(
        raw,
        question=question,
        task=task,
        thinking_language_target=target,
        requested_answer_language=answer_lang,
        n_tokens=n_tokens_i,
        gold=gold_s,
        options=options,
        answer_prefix=rec.get("answer_prefix"),
        in_language_threshold=threshold,
        extract=rec.get("extract"),
        max_tokens=rec.get("max_tokens"),
    )
    analysis = classify_record(
        {
            "thinking": score.split.thinking,
            "question": question,
            "user_message": rec.get("user_message") or question,
            "options": options,
            "loop": score.loop,
            "truncated": score.truncated,
        },
        target=target,
    )
    out = dict(rec)
    out.update(
        {
            "task": task,
            "thinking_language_target": target,
            "requested_answer_language": answer_lang,
            "raw_output": raw,
            "thinking": score.split.thinking,
            "answer_span": score.split.answer_span,
            "parsed_answer": score.parsed_answer,
            "parser_answer": score.parser_answer,
            "parser_correct": score.parser_correct,
            "thinking_lang_share": score.thinking_lang_share,
            "in_language_95": score.in_language_95,
            "answer_language": score.answer_language,
            "answer_lang_ok": score.answer_lang_ok,
            "closed": score.closed,
            "truncated": score.truncated,
            "loop": score.loop,
            "correct": score.correct,
            "n_tokens": n_tokens_i,
            "l2_primary": analysis["primary_with_closure"] if not score.in_language_95 else "pass",
            "l2_share_bucket": analysis["share_bucket"],
            "l2_scored_kinds": analysis["scored_kinds"],
            "l2_allowed_kinds": analysis["allowed_kinds"],
            "l2_leaks": analysis["leaks"],
        }
    )
    return out


def _resolve_task(task: Any, source: str) -> TaskKind | str:
    if task and task != "auto":
        return str(task)
    if source:
        try:
            return task_for_source(source)
        except ValueError:
            pass
    return "instruction"


def _raw_output(rec: Mapping[str, Any]) -> str:
    raw = rec.get("raw_output") or rec.get("raw") or rec.get("generation")
    if raw:
        return str(raw)
    thinking = str(rec.get("thinking") or "")
    answer = str(rec.get("answer_span") or rec.get("answer") or "")
    if thinking or answer:
        body = thinking
        if not thinking.startswith(THINK_OPEN):
            body = f"{THINK_OPEN}{thinking}"
        if THINK_CLOSE not in body:
            body = f"{body}{THINK_CLOSE}"
        return f"{body}\n{answer}".strip()
    return ""


def load_traces(path: Path, *, max_samples: int = 0) -> list[dict[str, Any]]:
    traces_file = path / "traces.jsonl" if path.is_dir() else path
    if not traces_file.is_file():
        raise FileNotFoundError(f"traces JSONL not found: {traces_file}")
    rows: list[dict[str, Any]] = []
    with traces_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def resolve_input(raw: Any, *, base: Path) -> Path:
    if not raw:
        raise ValueError("config.input is required (traces.jsonl or a directory containing it)")
    path = resolve_path(raw, base=base)
    if path.exists():
        return path
    bundled = STEP_DIR / "config" / Path(str(raw)).name
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"input not found: {raw} (looked at {path} and {bundled})")


def resolve_path(raw: Any, *, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    from_base = (base / path).resolve()
    if from_base.exists():
        return from_base
    from_step = (STEP_DIR / path).resolve()
    if from_step.exists():
        return from_step
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    return from_base

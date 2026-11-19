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

"""Generate + W1.1-score Native-reasoning benches against a hosted endpoint."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from nemotron.steps.eval.mr_bench.benchmarks import (
    BenchmarkSpec,
    default_prompt_style,
    spec_for,
)
from nemotron.steps.eval.mr_bench.endpoint import (
    Completion,
    EndpointError,
    OpenAIChatClient,
    detect_model_name,
    is_connection_error,
    normalize_model_url,
    resolve_api_key,
)
from nemotron.steps.eval.mr_bench.fetch import fetch_sources
from nemotron.steps.eval.mr_bench.mmlu_prox import (
    MMLU_PROX_MAX_TOKENS,
    MMLU_PROX_SHOTS,
    MMLU_PROX_TEMPERATURE,
    MMLU_PROX_TOP_P,
    format_mmlu_prox_prompt,
    is_mmlu_prox_benchmark,
    is_mmlu_prox_official,
    load_mmlu_prox_fewshots,
    mmlu_prox_lang,
    mmlu_prox_letter_extractor,
    mmlu_prox_stop,
    parse_mmlu_prox_official,
)
from nemotron.steps.eval.mr_bench.polymath import (
    POLYMATH_MAX_TOKENS,
    POLYMATH_TEMPERATURE,
    POLYMATH_TOP_K,
    POLYMATH_TOP_P,
    difficulty_weighted_acc,
    format_polymath_prompt,
    is_polymath_benchmark,
    is_polymath_official,
    math_equal,
    parse_polymath_official,
)
from nemotron.steps.eval.mr_bench.prompts import (
    VMLU_LEADERBOARD_SYSTEM,
    assistant_prefill,
    build_prompt,
    dataset_instruction,
    format_vmlu_leaderboard_user,
)
from nemotron.steps.eval.mr_bench.schema import sample_log
from nemotron.steps.eval.mr_bench.sources import load_eval_records
from nemotron.steps.eval.mr_bench.vmlu_csv import write_vmlu_submission_csv
from nemotron.steps.eval.mr_l2.constants import (
    GSM8K_HI_ANSWER_PREFIX,
    MGSM_EN_ANSWER_PREFIX,
    MGSM_VI_ANSWER_PREFIX,
    SAMPLING,
    bench_language,
)
from nemotron.steps.eval.mr_l2.lid import default_detector
from nemotron.steps.eval.mr_l2.score import score_trace
from nemotron.steps.eval.mr_l2.types import EvalMode, EvalRecord

LOGGER = logging.getLogger(__name__)
STEP_DIR = Path(__file__).resolve().parent


class ChatClient(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Completion: ...


@dataclass
class RunConfig:
    model_url: str
    benchmark: str
    model_name: str | None = None
    api_key: str | None = None
    mode: EvalMode = "free"
    data_root: Path = Path("data")
    out: Path | None = None
    max_samples: int | None = None
    concurrency: int = 4
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    timeout: float = 600.0
    resume: bool = False
    overwrite: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    thinking: bool | None = None
    thinking_language_target: str | None = None
    requested_answer_language: str | None = None
    dry_run: bool = False
    fail_fast: bool = False
    healthcheck: bool = True
    vllm_prefill: bool = False
    prompt_style: str | None = None
    apply_chat_template: bool | None = None
    thinking_prefix_override: str | None = None
    fewshot_by_stratum: dict[str, tuple] = field(default_factory=dict)
    fetch: bool = True


def config_from_mapping(cfg: Mapping[str, Any], *, config_path: Path | None = None) -> RunConfig:
    base = config_path.parent if config_path is not None else Path.cwd()
    data_root = Path(str(cfg.get("data_root") or "data"))
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    out_raw = cfg.get("output_dir") or cfg.get("out")
    out = Path(str(out_raw)) if out_raw else None
    if out is not None and not out.is_absolute():
        out = (base / out).resolve()
    extra = cfg.get("extra_body") or {}
    if isinstance(extra, str) and extra.strip():
        extra = json.loads(extra)
    model_url = str(cfg.get("model_url") or os.environ.get("EVAL_ENDPOINT_URL") or "")
    model_name = cfg.get("model_name") or os.environ.get("EVAL_MODEL_HANDLE")
    api_key = cfg.get("api_key")
    if api_key is None:
        key_name = cfg.get("api_key_name") or os.environ.get("EVAL_API_KEY_NAME")
        if key_name:
            api_key = os.environ.get(str(key_name))
    max_samples = cfg.get("max_samples") or cfg.get("limit_samples")
    return RunConfig(
        model_url=model_url,
        benchmark=str(cfg.get("benchmark") or "tiny"),
        model_name=None if model_name in (None, "") else str(model_name),
        api_key=None if not api_key else str(api_key),
        mode=str(cfg.get("mode") or "free"),  # type: ignore[arg-type]
        data_root=data_root,
        out=out,
        max_samples=None if max_samples in (None, "", 0, "0") else int(max_samples),
        concurrency=int(cfg.get("concurrency") or 4),
        max_tokens=None if cfg.get("max_tokens") in (None, "") else int(cfg["max_tokens"]),
        temperature=None if cfg.get("temperature") in (None, "") else float(cfg["temperature"]),
        top_p=None if cfg.get("top_p") in (None, "") else float(cfg["top_p"]),
        timeout=float(cfg.get("timeout") or 600.0),
        resume=bool(cfg.get("resume")),
        overwrite=bool(cfg.get("overwrite")),
        extra_body=dict(extra or {}),
        thinking=cfg.get("thinking"),
        thinking_language_target=cfg.get("thinking_language_target") or None,
        requested_answer_language=cfg.get("requested_answer_language") or None,
        dry_run=bool(cfg.get("dry_run")),
        fail_fast=bool(cfg.get("fail_fast")),
        healthcheck=cfg.get("healthcheck", True) not in {False, "false", "0", 0},
        vllm_prefill=bool(cfg.get("vllm_prefill")),
        prompt_style=cfg.get("prompt_style") or None,
        apply_chat_template=cfg.get("apply_chat_template"),
        thinking_prefix_override=cfg.get("thinking_prefix") or cfg.get("thinking_prefix_override"),
        fetch=cfg.get("fetch", True) not in {False, "false", "0", 0},
    )


def resolve_prompt_style(cfg: RunConfig, spec: BenchmarkSpec) -> str:
    if cfg.prompt_style:
        return cfg.prompt_style
    return default_prompt_style(spec.name)


def resolve_languages(cfg: RunConfig, spec: BenchmarkSpec) -> tuple[str, str]:
    target = cfg.thinking_language_target or bench_language(spec.sources[0] if spec.sources else spec.name)
    answer = cfg.requested_answer_language or target
    return target, answer


def effective_sampling(cfg: RunConfig, prompt_style: str) -> dict[str, float | int]:
    if prompt_style == "mmlu-prox":
        temperature, top_p, max_tokens = MMLU_PROX_TEMPERATURE, MMLU_PROX_TOP_P, MMLU_PROX_MAX_TOKENS
    elif prompt_style == "polymath":
        temperature, top_p, max_tokens = POLYMATH_TEMPERATURE, POLYMATH_TOP_P, POLYMATH_MAX_TOKENS
    else:
        temperature, top_p, max_tokens = (
            SAMPLING["temperature"],
            SAMPLING["top_p"],
            SAMPLING["max_tokens"],
        )
    return {
        "temperature": float(cfg.temperature if cfg.temperature is not None else temperature),
        "top_p": float(cfg.top_p if cfg.top_p is not None else top_p),
        "max_tokens": int(cfg.max_tokens if cfg.max_tokens is not None else max_tokens),
    }


def uses_chat_template(cfg: RunConfig) -> bool:
    if cfg.apply_chat_template is not None:
        return bool(cfg.apply_chat_template)
    return True


def official_thinking(cfg: RunConfig, prompt_style: str) -> bool | None:
    if cfg.thinking is not None:
        return bool(cfg.thinking)
    if prompt_style == "mmlu-prox":
        return False
    return None


def extra_body_for(cfg: RunConfig, spec: BenchmarkSpec, prompt_style: str) -> dict[str, Any]:
    extra = dict(cfg.extra_body)
    if uses_chat_template(cfg):
        thinking = official_thinking(cfg, prompt_style)
        if thinking:
            kwargs = dict(extra.get("chat_template_kwargs") or {})
            kwargs.setdefault("enable_thinking", True)
            extra["chat_template_kwargs"] = kwargs
        elif thinking is False:
            kwargs = dict(extra.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = False
            extra["chat_template_kwargs"] = kwargs
        if cfg.mode == "forced" and cfg.vllm_prefill:
            extra.setdefault("continue_final_message", True)
            extra.setdefault("add_generation_prompt", False)
    if prompt_style == "mmlu-prox":
        extra.setdefault("stop", list(mmlu_prox_stop(mmlu_prox_lang(spec.name))))
    if prompt_style == "polymath":
        extra.setdefault("top_k", POLYMATH_TOP_K)
    return extra


def build_messages(
    record: EvalRecord,
    cfg: RunConfig,
    *,
    prompt_style: str,
    thinking_language_target: str,
    requested_answer_language: str,
    sampling: dict[str, float | int],
) -> tuple[list[dict[str, str]], Any]:
    leaderboard = prompt_style == "vmlu-leaderboard" and record.source == "vmlu"
    mmlu_official = is_mmlu_prox_official(prompt_style, record.source)
    polymath_official = is_polymath_official(prompt_style, record.source)
    user_override = None
    if leaderboard:
        user_override = format_vmlu_leaderboard_user(record.question, record.options)
    elif mmlu_official:
        shots = cfg.fewshot_by_stratum.get(record.stratum) or ()
        user_override = format_mmlu_prox_prompt(record, shots)
    elif polymath_official:
        user_override = format_polymath_prompt(record.question, lang=record.prompt_language or "vi")
    instruction = None
    if user_override is None:
        instruction = dataset_instruction(record.source, record.instruction)
    prompt = build_prompt(
        question=record.question,
        mode=cfg.mode,
        thinking_language_target=thinking_language_target,
        requested_answer_language=requested_answer_language,
        options=record.options,
        user_message=user_override,
        sampling=sampling,
        thinking_prefix_override=cfg.thinking_prefix_override,
        instruction=instruction,
    )
    messages: list[dict[str, str]] = []
    if leaderboard:
        messages.append({"role": "system", "content": VMLU_LEADERBOARD_SYSTEM})
    messages.append({"role": "user", "content": prompt.user_message})
    if cfg.mode == "forced":
        messages.append(
            {
                "role": "assistant",
                "content": assistant_prefill(
                    cfg.mode,
                    thinking_language_target,
                    cfg.thinking_prefix_override,
                ),
            }
        )
    return messages, prompt


def _extract_for(prompt_style: str | None, source: str | None) -> str | None:
    if is_mmlu_prox_official(prompt_style, source):
        return f"mmlu_prox:{mmlu_prox_lang(source)}"
    if is_polymath_official(prompt_style, source):
        return "polymath"
    return None


def _apply_official_extract(
    row: dict[str, Any],
    record: EvalRecord,
    prompt_style: str,
    score_gold: bool,
) -> None:
    gold = record.gold or None
    if is_polymath_official(prompt_style, record.source):
        pred = parse_polymath_official(str(row.get("answer_span") or ""))
        row["parsed_answer"] = pred
        row["extract"] = "polymath_official"
        if score_gold and gold:
            row["correct"] = math_equal(pred, gold)
        return
    if is_mmlu_prox_official(prompt_style, record.source):
        lang = mmlu_prox_lang(record.source)
        pred = parse_mmlu_prox_official(str(row.get("answer_span") or ""), lang=lang) or parse_mmlu_prox_official(
            str(row.get("raw_output") or ""), lang=lang
        )
        row["parsed_answer"] = pred
        row["extract"] = f"mmlu_prox:{lang}"
        if score_gold and gold:
            row["correct"] = pred.strip().upper() == gold.strip().upper()[:1]


def _score_record(
    record: EvalRecord,
    raw: str,
    n_tokens: int | None,
    prompt: Any,
    cfg: RunConfig,
    spec: BenchmarkSpec,
    detector: Any,
    *,
    prompt_style: str,
    thinking_language_target: str,
    requested_answer_language: str,
    sampling: dict[str, float | int],
) -> dict[str, Any]:
    gold = record.gold if record.gold else None
    score_gold = spec.score_gold and gold is not None
    score = score_trace(
        raw,
        question=record.question,
        task=record.task,
        thinking_language_target=thinking_language_target,
        requested_answer_language=requested_answer_language,
        n_tokens=n_tokens,
        gold=gold if score_gold else None,
        options=record.options,
        detector=detector,
        path="eval",
        answer_prefix=record.answer_prefix,
        extract=_extract_for(prompt_style, record.source),
        max_tokens=int(sampling["max_tokens"]),
    )
    log = sample_log(
        item_id=record.item_id,
        source=record.source,
        split=record.split,
        mode=cfg.mode,
        path="eval",
        task=record.task,
        prompt_language=record.prompt_language,
        prompt=prompt,
        raw_output=raw,
        score=score,
        gold=gold if score_gold else None,
    )
    log["stratum"] = record.stratum
    if record.options is not None:
        log["options"] = list(record.options)
    _apply_official_extract(log, record, prompt_style, score_gold)
    return log


def aggregate_metrics(
    spec: BenchmarkSpec,
    records: Sequence[dict[str, Any]],
    *,
    model_url: str,
    model_name: str,
    mode: str,
    prompt_style: str,
) -> dict[str, Any]:
    n = len(records)
    errors = [r for r in records if r.get("error")]
    attempted = [r for r in records if not r.get("error")]
    scored = [r for r in attempted if r.get("correct") is not None]
    parsed = [r for r in attempted if str(r.get("parsed_answer") or "")]
    closed = [r for r in attempted if r.get("closed")]
    in_lang = [r for r in attempted if r.get("in_language_95")]
    n_ok = len(attempted)
    by_stratum: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = str(rec.get("stratum") or "unknown")
        bucket = by_stratum.setdefault(key, {"n": 0, "correct": 0, "scored": 0})
        bucket["n"] += 1
        if rec.get("error"):
            continue
        if rec.get("correct") is not None:
            bucket["scored"] += 1
            if rec.get("correct"):
                bucket["correct"] += 1
    for bucket in by_stratum.values():
        bucket["accuracy"] = (bucket["correct"] / bucket["scored"]) if bucket["scored"] else None
    metrics: dict[str, Any] = {
        "benchmark": spec.name,
        "model_url": model_url,
        "model_name": model_name,
        "mode": mode,
        "prompt_style": prompt_style,
        "n": n,
        "n_parsed": len(parsed),
        "parse_rate": (len(parsed) / n_ok) if n_ok else 0.0,
        "n_errors": len(errors),
        "n_attempted": n_ok,
        "closed_rate": (len(closed) / n_ok) if n_ok else 0.0,
        "in_language_95_rate": (len(in_lang) / n_ok) if n_ok else 0.0,
        "shots": MMLU_PROX_SHOTS if prompt_style == "mmlu-prox" else 0,
        "by_stratum": by_stratum,
        "w11_scores": True,
    }
    if spec.score_gold and scored:
        n_correct = sum(1 for r in scored if r.get("correct"))
        metrics["n_scored"] = len(scored)
        metrics["n_correct"] = n_correct
        metrics["accuracy"] = n_correct / len(scored)
    else:
        metrics["n_scored"] = 0
        metrics["accuracy"] = None
    parser_scored = [r for r in attempted if r.get("parser_correct") is not None]
    if parser_scored:
        n_parser_parsed = sum(1 for r in parser_scored if str(r.get("parser_answer") or ""))
        n_parser_ok = sum(1 for r in parser_scored if r.get("parser_correct"))
        prefix = (
            GSM8K_HI_ANSWER_PREFIX
            if spec.name in {"gsm8k_hi", "gsm8k_en"}
            else MGSM_EN_ANSWER_PREFIX
            if spec.name == "mgsm_en"
            else MGSM_VI_ANSWER_PREFIX
        )
        metrics["parser_metric"] = {
            "extractor": "mgsm_prefix",
            "prefix": prefix,
            "n_scored": len(parser_scored),
            "n_parsed": n_parser_parsed,
            "parse_rate": n_parser_parsed / len(parser_scored),
            "n_correct": n_parser_ok,
            "accuracy": n_parser_ok / len(parser_scored),
        }
    if is_polymath_benchmark(spec.name):
        dw = difficulty_weighted_acc(metrics.get("by_stratum") or {})
        metrics.update(dw)
        official = is_polymath_official(prompt_style, spec.name)
        metrics["match_metric"] = "math_equal" if official else "w11_match"
        metrics["extractor"] = "first_boxed_answer_span" if official else "last_boxed_or_number"
        metrics["official_dw_acc"] = official
        metrics["n_repeats"] = 1
        metrics["report"] = "pass@1"
    if is_mmlu_prox_official(prompt_style, spec.name):
        metrics["letter_extractor"] = mmlu_prox_letter_extractor(mmlu_prox_lang(spec.name))
    return metrics


def _rewrite_traces(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_existing_traces(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_id = rec.get("item_id")
            if item_id and rec.get("error") in (None, ""):
                out[str(item_id)] = rec
    return out


def select_records(cfg: RunConfig, spec: BenchmarkSpec) -> list[EvalRecord]:
    records = load_eval_records(cfg.data_root, sources=spec.sources, splits=spec.default_splits)
    if cfg.max_samples is not None and cfg.max_samples >= 0:
        records = records[: cfg.max_samples]
    if not records:
        raise FileNotFoundError(
            f"no {spec.name} items under {cfg.data_root} (sources={spec.sources}). "
            "Set fetch=true or place parquet/jsonl in data_root."
        )
    return list(records)


def run_benchmark(
    cfg: RunConfig,
    *,
    client: ChatClient | None = None,
) -> dict[str, Any]:
    spec = spec_for(cfg.benchmark)
    prompt_style = resolve_prompt_style(cfg, spec)
    thinking_language_target, requested_answer_language = resolve_languages(cfg, spec)
    cfg = replace(
        cfg,
        prompt_style=prompt_style,
        thinking_language_target=thinking_language_target,
        requested_answer_language=requested_answer_language,
    )
    if cfg.fetch and spec.name != "tiny":
        fetch_sources(spec.sources, cfg.data_root)
    if is_mmlu_prox_official(prompt_style, spec.sources[0] if spec.sources else None):
        cfg.fewshot_by_stratum = load_mmlu_prox_fewshots(
            str(cfg.data_root), MMLU_PROX_SHOTS, lang=mmlu_prox_lang(spec.name)
        )
    items = select_records(cfg, spec)
    sampling = effective_sampling(cfg, prompt_style)
    extra = extra_body_for(cfg, spec, prompt_style)

    if client is None:
        if not cfg.model_url:
            raise ValueError("model_url is required (or set EVAL_ENDPOINT_URL)")
        endpoint = normalize_model_url(cfg.model_url)
        api_key = resolve_api_key(cfg.api_key)
        if cfg.dry_run:
            model_name = cfg.model_name or "dry-run"
        else:
            model_name = detect_model_name(endpoint, api_key, cfg.model_name)
        client = OpenAIChatClient(
            endpoint,
            model_name,
            api_key=api_key,
            timeout=cfg.timeout,
            extra_body=extra,
        )
        if cfg.healthcheck and not cfg.dry_run:
            try:
                client.list_models()
            except Exception as exc:  # noqa: BLE001
                if is_connection_error(exc):
                    raise EndpointError(f"endpoint unreachable: {exc}") from exc
                LOGGER.warning("/v1/models healthcheck failed: %s", exc)
    else:
        model_name = cfg.model_name or "injected"

    out_dir = cfg.out or (Path("output") / "mr_bench" / spec.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    prompts_path = out_dir / "prompts.jsonl"
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model_url": cfg.model_url,
                "model_name": model_name,
                "benchmark": spec.name,
                "mode": cfg.mode,
                "n": len(items),
                "max_tokens": sampling["max_tokens"],
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "prompt_style": prompt_style,
                "thinking_language_target": thinking_language_target,
                "requested_answer_language": requested_answer_language,
                "vllm_prefill": cfg.vllm_prefill,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if cfg.dry_run:
        lines = []
        for rec in items:
            messages, prompt = build_messages(
                rec,
                cfg,
                prompt_style=prompt_style,
                thinking_language_target=thinking_language_target,
                requested_answer_language=requested_answer_language,
                sampling=sampling,
            )
            lines.append(
                json.dumps(
                    {
                        "item_id": rec.item_id,
                        "source": rec.source,
                        "task": rec.task,
                        "user_message": prompt.user_message,
                        "messages": messages,
                    },
                    ensure_ascii=False,
                )
            )
        prompts_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        metrics = {
            "benchmark": spec.name,
            "model_url": cfg.model_url,
            "model_name": model_name,
            "mode": cfg.mode,
            "n": len(items),
            "dry_run": True,
            "accuracy": None,
            "out_dir": str(out_dir),
            "prompts": str(prompts_path),
            "w11_scores": True,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("dry-run %s: %s prompts -> %s", spec.name, len(items), prompts_path)
        return metrics

    if traces_path.exists() and not cfg.resume and not cfg.overwrite:
        raise FileExistsError(f"{traces_path} exists; set resume=true or overwrite=true")
    existing = _load_existing_traces(traces_path) if cfg.resume else {}
    if cfg.overwrite and traces_path.exists():
        traces_path.unlink()
        existing = {}

    detector = default_detector()
    lock = threading.Lock()
    pending = [rec for rec in items if rec.item_id not in existing]
    total = len(items)

    def _write(row: dict[str, Any]) -> None:
        with lock:
            with traces_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _one(rec: EvalRecord) -> dict[str, Any]:
        messages, prompt = build_messages(
            rec,
            cfg,
            prompt_style=prompt_style,
            thinking_language_target=thinking_language_target,
            requested_answer_language=requested_answer_language,
            sampling=sampling,
        )
        try:
            completion = client.complete(
                messages,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                max_tokens=int(sampling["max_tokens"]),
            )
            row = _score_record(
                rec,
                completion.text,
                completion.n_tokens,
                prompt,
                cfg,
                spec,
                detector,
                prompt_style=prompt_style,
                thinking_language_target=thinking_language_target,
                requested_answer_language=requested_answer_language,
                sampling=sampling,
            )
        except Exception as exc:  # noqa: BLE001
            if cfg.fail_fast:
                raise
            row = _score_record(
                rec,
                "",
                None,
                prompt,
                cfg,
                spec,
                detector,
                prompt_style=prompt_style,
                thinking_language_target=thinking_language_target,
                requested_answer_language=requested_answer_language,
                sampling=sampling,
            )
            row["error"] = str(exc)
            row["raw_output"] = ""
            row["correct"] = None
            row["parsed_answer"] = ""
        _write(row)
        return row

    results = dict(existing)
    if pending:
        workers = max(1, min(int(cfg.concurrency), len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, rec): rec.item_id for rec in pending}
            for fut in as_completed(futures):
                item_id = futures[fut]
                row = fut.result()
                results[item_id] = row
                LOGGER.info("%s: %s/%s", spec.name, len(results), total)

    ordered = [results[r.item_id] for r in items if r.item_id in results]
    if ordered:
        _rewrite_traces(traces_path, ordered)
    metrics = aggregate_metrics(
        spec,
        ordered,
        model_url=cfg.model_url,
        model_name=model_name,
        mode=cfg.mode,
        prompt_style=prompt_style,
    )
    if spec.submission_csv:
        submission = write_vmlu_submission_csv(ordered, out_dir / "submission.csv")
        metrics["submission_csv"] = str(submission)
    metrics["out_dir"] = str(out_dir)
    metrics["traces"] = str(traces_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s n=%s acc=%s", out_dir, metrics.get("n"), metrics.get("accuracy"))
    return metrics


def run_mr_bench(cfg: Mapping[str, Any], *, config_path: Path | None = None, client: ChatClient | None = None) -> dict[str, Any]:
    run = config_from_mapping(cfg, config_path=config_path)
    return run_benchmark(run, client=client)

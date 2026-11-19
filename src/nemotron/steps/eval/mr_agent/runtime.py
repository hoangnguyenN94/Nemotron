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

"""Run SEATau-vi / τ-bench 2.0 HI/EN against a hosted agent + NVIDIA user-sim."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.eval.mr_agent.clone import prepare_harness
from nemotron.steps.eval.mr_agent.commands import CommandSpec, plan_runs, redacted
from nemotron.steps.eval.mr_agent.metrics import aggregate_domain_summaries, summarize_results
from nemotron.steps.eval.mr_agent.pins import (
    CS_OVERLAP_DOMAINS,
    DEFAULT_SEATAU_SCENARIO,
    DEFAULT_SEATAU_TRIALS,
    DEFAULT_TAU2_TRIALS,
    DEFAULT_USER_API_BASE,
    DEFAULT_USER_MODEL,
    SEATAU_DOMAINS,
    SEATAU_SCENARIOS,
    TAU2_DOMAINS,
)
from nemotron.steps.eval.mr_l2.constants import FORCED_OPENERS

LOGGER = logging.getLogger(__name__)

SUITES = {
    "tiny": {"harness": "seatau", "language": "vi", "domains": ("retail",), "scenario": DEFAULT_SEATAU_SCENARIO},
    "seatau_vi": {
        "harness": "seatau",
        "language": "vi",
        "domains": SEATAU_DOMAINS,
        "scenario": DEFAULT_SEATAU_SCENARIO,
    },
    "tau2_hi": {"harness": "tau2", "language": "hi", "domains": TAU2_DOMAINS, "scenario": None},
    "tau2_en": {"harness": "tau2", "language": "en", "domains": TAU2_DOMAINS, "scenario": None},
}


@dataclass
class RunConfig:
    suite: str
    model_url: str
    model_name: str | None = None
    mode: str = "free"
    data_root: Path = Path("data")
    out: Path | None = None
    domains: tuple[str, ...] | None = None
    scenario: str | None = None
    num_trials: int | None = None
    num_tasks: int | None = None
    task_ids: tuple[str, ...] = ()
    max_concurrency: int = 1
    timeout: float | None = None
    thinking_prefix: str | None = None
    vllm_prefill: bool | None = None
    enable_thinking: bool = True
    agent_max_tokens: int | None = None
    agent_api_key: str = "EMPTY"
    user_model: str | None = None
    user_api_base: str | None = None
    api_key_name: str = "NGC_API_KEY"
    tag: str = "mr-agent"
    fetch: bool = True
    overlay: bool = True
    sync: bool = True
    dry_run: bool = False
    auto_resume: bool = True
    keep_going: bool = True
    log_level: str = "INFO"


def _as_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    return value not in {False, "false", "0", 0}


def _split_csv(raw: Any) -> tuple[str, ...] | None:
    if raw in (None, "", []):
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw if str(x).strip())
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def config_from_mapping(cfg: Mapping[str, Any], *, config_path: Path | None = None) -> RunConfig:
    base = config_path.parent if config_path is not None else Path.cwd()
    data_root = Path(str(cfg.get("data_root") or cfg.get("harness_root") or "data"))
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    out_raw = cfg.get("output_dir") or cfg.get("out")
    out = Path(str(out_raw)) if out_raw else None
    if out is not None and not out.is_absolute():
        out = (base / out).resolve()
    max_tasks = cfg.get("num_tasks") or cfg.get("max_samples") or cfg.get("limit_samples")
    return RunConfig(
        suite=str(cfg.get("benchmark") or cfg.get("suite") or "tiny"),
        model_url=str(cfg.get("model_url") or os.environ.get("EVAL_ENDPOINT_URL") or ""),
        model_name=(cfg.get("model_name") or os.environ.get("EVAL_MODEL_HANDLE") or None),
        mode=str(cfg.get("mode") or "free"),
        data_root=data_root,
        out=out,
        domains=_split_csv(cfg.get("domains") or cfg.get("domain")),
        scenario=cfg.get("scenario") or None,
        num_trials=None if cfg.get("num_trials") in (None, "") else int(cfg["num_trials"]),
        num_tasks=None if max_tasks in (None, "", 0, "0") else int(max_tasks),
        task_ids=tuple(_split_csv(cfg.get("task_ids")) or ()),
        max_concurrency=int(cfg.get("max_concurrency") or cfg.get("concurrency") or 1),
        timeout=None if cfg.get("timeout") in (None, "") else float(cfg["timeout"]),
        thinking_prefix=cfg.get("thinking_prefix") or None,
        vllm_prefill=None if cfg.get("vllm_prefill") in (None, "") else _as_bool(cfg.get("vllm_prefill"), True),
        enable_thinking=_as_bool(cfg.get("enable_thinking"), True),
        agent_max_tokens=None if cfg.get("agent_max_tokens") in (None, "") else int(cfg["agent_max_tokens"]),
        agent_api_key=str(cfg.get("agent_api_key") or os.environ.get("SEATAU_AGENT_API_KEY") or "EMPTY"),
        user_model=cfg.get("user_model") or os.environ.get("SEATAU_USER_MODEL") or os.environ.get("TAU2_USER_MODEL"),
        user_api_base=cfg.get("user_api_base")
        or os.environ.get("SEATAU_USER_API_BASE")
        or os.environ.get("TAU2_USER_API_BASE"),
        api_key_name=str(cfg.get("api_key_name") or os.environ.get("EVAL_API_KEY_NAME") or "NGC_API_KEY"),
        tag=str(cfg.get("agent_tag") or cfg.get("tag") or "mr-agent"),
        fetch=_as_bool(cfg.get("fetch"), True),
        overlay=_as_bool(cfg.get("overlay"), True),
        sync=_as_bool(cfg.get("sync"), True),
        dry_run=_as_bool(cfg.get("dry_run"), False),
        auto_resume=_as_bool(cfg.get("auto_resume"), True),
        keep_going=_as_bool(cfg.get("keep_going"), True),
        log_level=str(cfg.get("log_level") or "INFO"),
    )


def resolve_suite(cfg: RunConfig) -> dict[str, Any]:
    key = cfg.suite.strip().lower().replace("-", "_")
    aliases = {"seatau": "seatau_vi", "tau2-hi": "tau2_hi", "tau2-en": "tau2_en", "tau_bench_hi": "tau2_hi"}
    key = aliases.get(key, key)
    if key not in SUITES:
        raise ValueError(f"unknown suite {cfg.suite!r}. Choose from: {', '.join(SUITES)}")
    spec = dict(SUITES[key])
    spec["suite"] = key
    if cfg.domains:
        spec["domains"] = cfg.domains
    if spec["harness"] == "seatau":
        if cfg.scenario:
            spec["scenario"] = cfg.scenario
        if spec.get("scenario") not in SEATAU_SCENARIOS:
            raise ValueError(f"unknown SEATau scenario {spec.get('scenario')!r}. Choose from: {SEATAU_SCENARIOS}")
    else:
        spec["scenario"] = None
    return spec


def _normalize_url(url: str) -> str:
    text = (url or "").rstrip("/")
    if text.endswith("/chat/completions"):
        text = text[: -len("/chat/completions")]
    return text


def _write_dotenv(clone_root: Path, *, user_base: str, user_model: str, agent_base: str, agent_model: str, agent_key: str) -> None:
    lines = [
        f"NGC_API_KEY={os.environ.get('NGC_API_KEY', '')}",
        f"SEATAU_USER_API_BASE={user_base}",
        f"SEATAU_USER_MODEL={user_model}",
        f"SEATAU_JUDGE_MODEL={os.environ.get('SEATAU_JUDGE_MODEL') or user_model}",
        f"SEATAU_AGENT_API_BASE={agent_base}",
        f"SEATAU_AGENT_MODEL={agent_model}",
        f"SEATAU_AGENT_API_KEY={agent_key}",
        f"TAU2_USER_API_BASE={user_base}",
        f"TAU2_USER_MODEL={user_model}",
        f"TAU2_JUDGE_MODEL={os.environ.get('SEATAU_JUDGE_MODEL') or os.environ.get('TAU2_JUDGE_MODEL') or user_model}",
        "TAU2_FASTTEXT_LID_MODEL_PATH=data/models/lid.176.bin",
    ]
    (clone_root / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_tree(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return
    shutil.copytree(src, dest, dirs_exist_ok=True)


def build_spec(cfg: RunConfig) -> CommandSpec:
    suite = resolve_suite(cfg)
    language = str(suite["language"])
    mode = cfg.mode if cfg.mode in {"free", "forced"} else "free"
    prefix = cfg.thinking_prefix
    if mode == "forced" and not prefix:
        prefix = FORCED_OPENERS.get(language, FORCED_OPENERS["en"])
    if mode == "forced":
        vllm_prefill = True if cfg.vllm_prefill is None else bool(cfg.vllm_prefill)
    else:
        vllm_prefill = bool(cfg.vllm_prefill)
    trials = cfg.num_trials
    if trials is None:
        trials = DEFAULT_SEATAU_TRIALS if suite["harness"] == "seatau" else DEFAULT_TAU2_TRIALS
    if suite["suite"] == "tiny":
        trials = cfg.num_trials if cfg.num_trials is not None else 1
    agent_base = _normalize_url(cfg.model_url or os.environ.get("SEATAU_AGENT_API_BASE") or "")
    agent_llm = cfg.model_name or os.environ.get("SEATAU_AGENT_MODEL") or "dummy"
    user_base = _normalize_url(cfg.user_api_base or os.environ.get("SEATAU_USER_API_BASE") or DEFAULT_USER_API_BASE)
    user_llm = cfg.user_model or DEFAULT_USER_MODEL
    return CommandSpec(
        suite=str(suite["suite"]),
        harness=str(suite["harness"]),
        language=language,
        domains=tuple(suite["domains"]),
        scenario=suite.get("scenario"),
        mode=mode,
        agent_llm=agent_llm,
        agent_api_base=agent_base or "http://127.0.0.1:8000/v1",
        agent_api_key=cfg.agent_api_key,
        user_llm=user_llm,
        user_api_base=user_base,
        num_trials=int(trials),
        max_concurrency=cfg.max_concurrency,
        tag=cfg.tag,
        thinking_prefix=prefix if mode == "forced" else None,
        vllm_prefill=bool(vllm_prefill) if mode == "forced" else False,
        enable_thinking=cfg.enable_thinking or mode in {"free", "forced"},
        agent_max_tokens=cfg.agent_max_tokens,
        num_tasks=1 if suite["suite"] == "tiny" and cfg.num_tasks is None else cfg.num_tasks,
        task_ids=cfg.task_ids,
        auto_resume=cfg.auto_resume,
        log_level=cfg.log_level,
        timeout=cfg.timeout,
    )


def run_benchmark(cfg: RunConfig) -> dict[str, Any]:
    spec = build_spec(cfg)
    runs = plan_runs(spec)
    out_dir = cfg.out or (Path("output") / "mr_agent" / spec.suite)
    out_dir.mkdir(parents=True, exist_ok=True)
    harness_root = cfg.data_root / spec.harness
    plan_path = out_dir / "commands.json"
    plan_path.write_text(
        json.dumps(
            {
                "suite": spec.suite,
                "harness": spec.harness,
                "mode": spec.mode,
                "language": spec.language,
                "domains": list(spec.domains),
                "scenario": spec.scenario,
                "num_trials": spec.num_trials,
                "cs_overlap_domains": list(CS_OVERLAP_DOMAINS),
                "runs": [
                    {
                        "domain": run.domain,
                        "save_to": run.save_to,
                        "cmd": redacted(run.cmd),
                    }
                    for run in runs
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if cfg.dry_run:
        metrics = {
            "suite": spec.suite,
            "mode": spec.mode,
            "dry_run": True,
            "n_runs": len(runs),
            "domains": list(spec.domains),
            "out_dir": str(out_dir),
            "commands": str(plan_path),
            "w11_scores": True,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("dry-run %s: %s commands -> %s", spec.suite, len(runs), plan_path)
        return metrics

    if not spec.agent_api_base or spec.agent_llm in {None, "", "dummy"}:
        raise ValueError("model_url and model_name are required (or EVAL_ENDPOINT_URL / EVAL_MODEL_HANDLE)")
    if shutil.which("uv") is None:
        raise RuntimeError("uv is not on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
    if shutil.which("git") is None:
        raise RuntimeError("git is not on PATH")
    key = os.environ.get(cfg.api_key_name) or os.environ.get("NGC_API_KEY")
    if not key:
        raise RuntimeError(f"{cfg.api_key_name} (user-sim / GPT-5.5) is empty")
    os.environ.setdefault("NGC_API_KEY", key)

    if cfg.fetch:
        prepare_harness(spec.harness, harness_root, overlay=cfg.overlay, lid=spec.harness == "seatau")
    elif not (harness_root / "src" / "tau2").is_dir():
        raise FileNotFoundError(f"harness missing at {harness_root}. Set fetch=true.")
    elif cfg.overlay:
        from nemotron.steps.eval.mr_agent.overlay import apply_harness_overlay

        apply_harness_overlay(harness_root, harness=spec.harness)

    _write_dotenv(
        harness_root,
        user_base=spec.user_api_base,
        user_model=spec.user_llm,
        agent_base=spec.agent_api_base,
        agent_model=spec.agent_llm,
        agent_key=spec.agent_api_key,
    )
    os.environ["SEATAU_SKIP_DOTENV"] = "1"
    if spec.harness == "seatau":
        os.environ["SEATAU_ROOT"] = str(harness_root)
        os.environ["TAU2_FASTTEXT_LID_MODEL_PATH"] = str(harness_root / "data" / "models" / "lid.176.bin")

    if cfg.sync:
        extras = ["uv", "sync"]
        if spec.harness == "seatau":
            extras.extend(["--extra", "experiments", "--extra", "dev"])
        LOGGER.info("uv sync in %s", harness_root)
        subprocess.check_call(extras, cwd=harness_root)

    failed: list[str] = []
    for run in runs:
        LOGGER.info("%s", redacted(run.cmd))
        rc = subprocess.call(run.cmd, cwd=harness_root, env=os.environ.copy())
        if rc != 0:
            failed.append(f"{run.domain}:{rc}")
            if not cfg.keep_going:
                raise RuntimeError(f"tau2 failed for {run.domain} (exit {rc})")

    sim_root = harness_root / "data" / "simulations"
    copied = out_dir / "simulations"
    _copy_tree(sim_root, copied)
    rows = []
    for run in runs:
        path = copied / run.save_to / "results.json"
        if not path.is_file():
            path = sim_root / run.save_to / "results.json"
        row = summarize_results(path)
        row["save_to"] = run.save_to
        row["domain"] = row.get("domain") or run.domain
        rows.append(row)
    metrics = aggregate_domain_summaries(rows, suite=spec.suite, mode=spec.mode)
    metrics["out_dir"] = str(out_dir)
    metrics["harness_root"] = str(harness_root)
    metrics["failed"] = failed
    if failed:
        metrics["error"] = "tau2_failed " + " ".join(failed)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s pass1=%s", out_dir, metrics.get("pass1"))
    return metrics


def run_mr_agent(cfg: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    return run_benchmark(config_from_mapping(cfg, config_path=config_path))

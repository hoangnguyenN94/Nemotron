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

"""Build `uv run tau2` argv. Does not execute."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nemotron.steps.eval.mr_agent.pins import INTAKE_ELICITATION_STRATEGY
from nemotron.steps.eval.mr_l2.constants import FORCED_OPENERS


@dataclass(frozen=True)
class AgentRun:
    suite: str
    harness: str
    language: str
    domain: str
    scenario: str | None
    save_to: str
    cmd: list[str]


def llm_args_json(
    api_base: str,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
    extra: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> str:
    payload: dict[str, Any] = {"api_base": api_base, "temperature": temperature}
    if api_key_env:
        payload["api_key_env"] = api_key_env
    elif api_key:
        payload["api_key"] = api_key
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if extra:
        payload.update(extra)
    return json.dumps(payload, separators=(",", ":"))


def agent_extra(
    *,
    mode: str,
    thinking_prefix: str | None,
    vllm_prefill: bool,
    enable_thinking: bool,
) -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    if mode == "forced" and thinking_prefix:
        extra["thinking_prefix"] = thinking_prefix
        extra["vllm_prefill"] = bool(vllm_prefill)
        enable_thinking = True
    if enable_thinking or extra.get("thinking_prefix"):
        extra_body = dict(extra.get("extra_body") or {})
        chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault("enable_thinking", True)
        extra_body["chat_template_kwargs"] = chat_kwargs
        extra["extra_body"] = extra_body
    return extra or None


def forced_opener(language: str, override: str | None = None) -> str:
    if override:
        return override
    return FORCED_OPENERS.get(language, FORCED_OPENERS["en"])


def save_name(
    *,
    tag: str,
    language: str,
    domain: str,
    scenario: str | None,
    mode: str,
) -> str:
    suffix = "_forced" if mode == "forced" else "_thinking"
    if scenario:
        return f"{tag}_{language}_{scenario}_{domain}{suffix}"
    return f"{tag}_{language}_{domain}{suffix}"


def redacted(cmd: Sequence[str]) -> str:
    out: list[str] = []
    hide_next = False
    for part in cmd:
        if hide_next:
            out.append("<json>")
            hide_next = False
            continue
        if part in {"--agent-llm-args", "--user-llm-args", "--communicate-judge"}:
            hide_next = True
        out.append(part)
    return " ".join(out)


@dataclass
class CommandSpec:
    suite: str
    harness: str
    language: str
    domains: tuple[str, ...]
    scenario: str | None
    mode: str
    agent_llm: str
    agent_api_base: str
    agent_api_key: str
    user_llm: str
    user_api_base: str
    num_trials: int
    max_concurrency: int
    tag: str
    thinking_prefix: str | None = None
    vllm_prefill: bool = False
    enable_thinking: bool = True
    agent_max_tokens: int | None = None
    num_tasks: int | None = None
    task_ids: tuple[str, ...] = ()
    auto_resume: bool = True
    log_level: str = "INFO"
    verbose_logs: bool = False
    timeout: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def seatau_cmd(spec: CommandSpec, domain: str) -> list[str]:
    extra = agent_extra(
        mode=spec.mode,
        thinking_prefix=spec.thinking_prefix,
        vllm_prefill=spec.vllm_prefill,
        enable_thinking=spec.enable_thinking,
    )
    lang_id = "en" if spec.scenario == "english" else spec.language
    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        domain,
        "--seatau-scenario",
        spec.scenario or "l2_interaction",
        "--lang-id",
        lang_id,
        "--agent-llm",
        spec.agent_llm,
        "--agent-llm-args",
        llm_args_json(
            spec.agent_api_base,
            api_key=spec.agent_api_key,
            extra=extra,
            max_tokens=spec.agent_max_tokens,
        ),
        "--user-llm",
        spec.user_llm,
        "--user-llm-args",
        llm_args_json(spec.user_api_base, api_key_env="NGC_API_KEY"),
        "--num-trials",
        str(spec.num_trials),
        "--max-concurrency",
        str(spec.max_concurrency),
        "--save-to",
        save_name(
            tag=spec.tag,
            language=spec.language,
            domain=domain,
            scenario=spec.scenario,
            mode=spec.mode,
        ),
        "--log-level",
        spec.log_level,
    ]
    if spec.num_tasks is not None:
        cmd.extend(["--num-tasks", str(spec.num_tasks)])
    if spec.task_ids:
        cmd.extend(["--task-ids", *spec.task_ids])
    if spec.verbose_logs:
        cmd.append("--verbose-logs")
    if spec.auto_resume:
        cmd.append("--auto-resume")
    return cmd


def tau2_cmd(spec: CommandSpec, domain: str) -> list[str]:
    extra = agent_extra(
        mode=spec.mode,
        thinking_prefix=spec.thinking_prefix,
        vllm_prefill=spec.vllm_prefill,
        enable_thinking=spec.enable_thinking,
    )
    user_extra = {"reasoning_effort": "low"}
    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--benchmark",
        "none",
        "--language",
        spec.language,
        "--domain",
        domain,
        "--agent-llm",
        spec.agent_llm,
        "--agent-llm-args",
        llm_args_json(
            spec.agent_api_base,
            api_key=spec.agent_api_key,
            extra=extra,
            max_tokens=spec.agent_max_tokens,
        ),
        "--user-llm",
        spec.user_llm,
        "--user-llm-args",
        llm_args_json(spec.user_api_base, api_key_env="NGC_API_KEY", extra=user_extra),
        "--num-trials",
        str(spec.num_trials),
        "--max-concurrency",
        str(spec.max_concurrency),
        "--save-to",
        save_name(
            tag=spec.tag,
            language=spec.language,
            domain=domain,
            scenario=None,
            mode=spec.mode,
        ),
        "--log-level",
        spec.log_level,
    ]
    if domain == "intake":
        cmd.extend(["--elicitation-strategy", INTAKE_ELICITATION_STRATEGY])
    if spec.timeout is not None:
        cmd.extend(["--timeout", str(spec.timeout)])
    if spec.num_tasks is not None:
        cmd.extend(["--num-tasks", str(spec.num_tasks)])
    if spec.task_ids:
        cmd.extend(["--task-ids", *spec.task_ids])
    if spec.verbose_logs:
        cmd.append("--verbose-logs")
    if spec.auto_resume:
        cmd.append("--auto-resume")
    return cmd


def plan_runs(spec: CommandSpec) -> list[AgentRun]:
    runs: list[AgentRun] = []
    builder = seatau_cmd if spec.harness == "seatau" else tau2_cmd
    for domain in spec.domains:
        cmd = builder(spec, domain)
        runs.append(
            AgentRun(
                suite=spec.suite,
                harness=spec.harness,
                language=spec.language,
                domain=domain,
                scenario=spec.scenario,
                save_to=save_name(
                    tag=spec.tag,
                    language=spec.language,
                    domain=domain,
                    scenario=spec.scenario,
                    mode=spec.mode,
                ),
                cmd=cmd,
            )
        )
    return runs

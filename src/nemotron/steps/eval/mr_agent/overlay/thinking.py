"""Forced-think + vLLM prefill helpers copied into a cloned tau2/SEATau tree.

This file is dropped onto `src/tau2/agent/thinking_overlay.py` after clone.
It must import only tau2 (not nemotron).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from tau2.data_model.message import AssistantMessage

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def thinking_prefill_content(prefix: str) -> str:
    """Match Native-reasoning forced mode: opener already inside the think block."""
    return f"{THINK_OPEN}{prefix}\n"


def prepare_thinking_generate(
    llm_args: Optional[dict],
) -> tuple[Optional[AssistantMessage], dict[str, Any]]:
    """Strip local thinking keys and optionally append a vLLM assistant prefill."""
    args: dict[str, Any] = deepcopy(llm_args) if llm_args else {}
    prefix = args.pop("thinking_prefix", None)
    vllm_prefill = bool(args.pop("vllm_prefill", False))
    if not prefix:
        return None, args
    prefill = AssistantMessage(role="assistant", content=thinking_prefill_content(str(prefix)))
    if vllm_prefill:
        extra_body = dict(args.get("extra_body") or {})
        extra_body.setdefault("continue_final_message", True)
        extra_body.setdefault("add_generation_prompt", False)
        chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault("enable_thinking", True)
        extra_body["chat_template_kwargs"] = chat_kwargs
        args["extra_body"] = extra_body
    return prefill, args


def stitch_thinking_prefill(
    assistant_message: AssistantMessage, prefill: Optional[AssistantMessage]
) -> AssistantMessage:
    """vLLM continue_final_message returns only new tokens; restore the forced opener."""
    if prefill is None or not prefill.content:
        return assistant_message
    content = assistant_message.content or ""
    if not content.strip():
        return assistant_message
    if content.startswith(THINK_OPEN):
        return assistant_message
    assistant_message.content = prefill.content + content
    return assistant_message


def reasoning_from_raw(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message") or {}
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("reasoning") or msg.get("reasoning_content") or "").strip()


def fold_reasoning(assistant_message: AssistantMessage) -> AssistantMessage:
    """Free-think often puts tokens in `reasoning`; tau2 only scores `content`."""
    content = assistant_message.content or ""
    if THINK_OPEN in content or THINK_CLOSE in content:
        return assistant_message
    reasoning = reasoning_from_raw(assistant_message.raw_data)
    if not reasoning:
        return assistant_message
    folded = f"{THINK_OPEN}{reasoning}{THINK_CLOSE}"
    if content.strip():
        assistant_message.content = f"{folded}\n{content}"
    else:
        assistant_message.content = folded
    return assistant_message

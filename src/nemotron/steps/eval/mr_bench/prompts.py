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

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from nemotron.steps.eval.mr_l2.constants import (
    FORCED_OPENERS,
    GSM8K_EN_INSTRUCTION,
    GSM8K_HI_INSTRUCTION,
    MCQ_AE,
    MCQ_AJ,
    MGSM_EN_INSTRUCTION,
    MGSM_VI_INSTRUCTION,
    MILU_1_5_INSTRUCTION,
    MMLU_PROX_EN_INSTRUCTION,
    MMLU_PROX_HI_INSTRUCTION,
    MMLU_PROX_VI_INSTRUCTION,
    SAMPLING,
    THINK_OPEN,
)
from nemotron.steps.eval.mr_l2.types import EvalMode, PromptRequest

VMLU_LEADERBOARD_SYSTEM = (
    "Bạn là một trợ lý thông minh. Hãy trả lời câu hỏi trắc nghiệm "
    "bằng cách chọn một trong các đáp án được liệt kê (A, B, C, D hoặc E). "
    "Chỉ trả lời bằng chữ cái đáp án, không cần giải thích thêm."
)


def format_user_message(
    question: str,
    options: Sequence[str | None] | None = None,
    instruction: str | None = None,
) -> str:
    """Free/forced user text: optional dataset instruction + question (+ options).

    Global-MGSM official concat is ``instruction + "\\n\\n" + question`` with no
    ``strip()`` — trailing space/newline on the question must be kept.
    """
    q = question or ""
    if options:
        lines = [q.strip(), ""]
        for letter, opt in zip(MCQ_AJ, options):
            if opt is None:
                continue
            text = str(opt).strip()
            if not text:
                continue
            lines.append(f"{letter}. {text}")
        body = "\n".join(lines).strip()
    else:
        body = q
    instr = (instruction or "").strip()
    if instr:
        return f"{instr}\n\n{body}"
    return body


def mgsm_instruction(explicit: str | None = None, lang: str = "vi") -> str:
    text = (explicit or "").strip()
    if text:
        return text
    return MGSM_EN_INSTRUCTION if lang == "en" else MGSM_VI_INSTRUCTION


def gsm8k_hi_instruction(explicit: str | None = None) -> str:
    text = (explicit or "").strip()
    return text or GSM8K_HI_INSTRUCTION


def gsm8k_en_instruction(explicit: str | None = None) -> str:
    text = (explicit or "").strip()
    return text or GSM8K_EN_INSTRUCTION


def mmlu_prox_instruction(explicit: str | None = None, lang: str = "vi") -> str:
    text = (explicit or "").strip()
    if text:
        return text
    if lang == "hi":
        return MMLU_PROX_HI_INSTRUCTION
    if lang == "en":
        return MMLU_PROX_EN_INSTRUCTION
    return MMLU_PROX_VI_INSTRUCTION


def milu_instruction(explicit: str | None = None) -> str:
    text = (explicit or "").strip()
    return text or MILU_1_5_INSTRUCTION


def dataset_instruction(source: str | None, explicit: str | None = None) -> str | None:
    if source == "mgsm_vi":
        return mgsm_instruction(explicit, lang="vi")
    if source == "mgsm_en":
        return mgsm_instruction(explicit, lang="en")
    if source == "gsm8k_hi":
        return gsm8k_hi_instruction(explicit)
    if source == "gsm8k_en":
        return gsm8k_en_instruction(explicit)
    if source in {"mmlu_prox_vi", "mmlu_prox_vi_val"}:
        return mmlu_prox_instruction(explicit, lang="vi")
    if source in {"mmlu_prox_hi", "mmlu_prox_hi_val"}:
        return mmlu_prox_instruction(explicit, lang="hi")
    if source in {"mmlu_prox_en", "mmlu_prox_en_val"}:
        return mmlu_prox_instruction(explicit, lang="en")
    if source == "milu_1_5":
        return milu_instruction(explicit)
    if source in {"ifeval", "ifeval_en"}:
        return None
    text = (explicit or "").strip()
    return text or None


def thinking_prefix(
    mode: EvalMode,
    thinking_language_target: str,
    override: str | None = None,
) -> str:
    if mode == "free":
        return ""
    if override is not None:
        return override
    try:
        return FORCED_OPENERS[thinking_language_target]
    except KeyError as exc:
        raise ValueError(f"no forced opener for {thinking_language_target!r}") from exc


def sampling_hash(sampling: dict[str, float | int] | None = None) -> str:
    payload = json.dumps(sampling or SAMPLING, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def format_vmlu_leaderboard_user(question: str, options: Sequence[str | None] | None = None) -> str:
    """Speaker / vmlu.ai-style user turn (not the W1.1 free estimand)."""
    letters: list[str] = []
    lines: list[str] = []
    for letter, opt in zip(MCQ_AE, options or ()):
        if opt is None:
            continue
        text = str(opt).strip()
        if not text:
            continue
        if not text.upper().startswith(f"{letter}."):
            text = f"{letter}. {text}"
        letters.append(letter)
        lines.append(text)
    last = letters[-1] if letters else "D"
    head = ", ".join(letters[:-1])
    valid = f"{head} hoặc {last}" if head else last
    return (
        f"Câu hỏi: {(question or '').strip()}\n"
        f"{chr(10).join(lines)}\n\n"
        f"Hãy chọn đáp án đúng ({valid}).\nCâu trả lời:"
    )


def build_prompt(
    *,
    question: str,
    mode: EvalMode,
    thinking_language_target: str,
    requested_answer_language: str,
    options: Sequence[str | None] | None = None,
    user_message: str | None = None,
    sampling: dict[str, float | int] | None = None,
    thinking_prefix_override: str | None = None,
    instruction: str | None = None,
) -> PromptRequest:
    user = (
        user_message
        if user_message is not None
        else format_user_message(question, options, instruction=instruction)
    )
    prefix = thinking_prefix(mode, thinking_language_target, thinking_prefix_override)
    return PromptRequest(
        mode=mode,
        user_message=user,
        thinking_prefix=prefix,
        thinking_language_target=thinking_language_target,
        requested_answer_language=requested_answer_language,
        sampling=dict(sampling or SAMPLING),
    )


def assistant_prefill(
    mode: EvalMode,
    thinking_language_target: str,
    override: str | None = None,
) -> str:
    """Assistant content continued with vLLM ``continue_final_message``.

    Nemotron-3.5 Lightning's chat template starts a thinking turn as
    ``<|im_start|>assistant\\n<think>\\n`` (prefix) and never prefills
    ``</think>`` (suffix — the model must emit it before the answer).
    The forced opener belongs *inside* that think block, after the newline.
    Glueing the opener onto ``<think>`` (no newline) leaves the reasoning
    parser stuck in-think, so ``Đáp án:`` never reaches ``content``.
    """
    if mode == "free":
        return f"{THINK_OPEN}\n"
    opener = thinking_prefix(mode, thinking_language_target, override)
    return f"{THINK_OPEN}\n{opener}\n"

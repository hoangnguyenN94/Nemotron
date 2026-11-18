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

import re
from typing import Sequence

from nemotron.steps.eval.mr_l2.constants import (
    MCQ_AE,
    MCQ_AJ,
    MGSM_VI_ANSWER_PREFIX,
    THINK_CLOSE,
    THINK_OPEN,
)
from nemotron.steps.eval.mr_l2.textnorm import strip_vietnamese_diacritics
from nemotron.steps.eval.mr_l2.types import TaskKind, ThinkAnswerSplit

_NUM_TOKEN = re.compile(
    r"[-+]?(?:"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+[.,]\d+"
    r"|\d+"
    r")"
)
_ANSWER_PREFIXES = (
    "đáp án chính xác nhất là",
    "dap an chinh xac nhat la",
    "đáp án chính xác là",
    "dap an chinh xac la",
    "đáp án đúng là",
    "dap an dung la",
    "đáp án là",
    "dap an la",
    "đáp án:",
    "dap an:",
    "đáp số:",
    "dap so:",
    "answer:",
    "the answer is",
    "final answer is",
    "####",
    "उत्तर:",
    "उत्तर है",
    "अंतिम उत्तर:",
    "câu trả lời là",
    "cau tra loi la",
    "trả lời:",
    "tra loi:",
)
_LETTER = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def split_think_answer(raw: str) -> ThinkAnswerSplit:
    """Split a Nano trace. Open tag may live only in the prompt."""
    text = raw or ""
    open_present = THINK_OPEN in text
    close_present = THINK_CLOSE in text

    if close_present:
        before, _, after = text.rpartition(THINK_CLOSE)
        if open_present:
            _, _, thinking = before.partition(THINK_OPEN)
        else:
            thinking = before
        return ThinkAnswerSplit(
            thinking=thinking.strip(),
            answer_span=after.strip(),
            think_open_present=open_present,
            think_close_present=True,
        )

    if open_present:
        _, _, thinking = text.partition(THINK_OPEN)
        return ThinkAnswerSplit(
            thinking=thinking.strip(),
            answer_span="",
            think_open_present=True,
            think_close_present=False,
        )

    return ThinkAnswerSplit(
        thinking=text.strip(),
        answer_span="",
        think_open_present=False,
        think_close_present=False,
    )


def extract_boxed(text: str) -> str | None:
    if not text or "boxed" not in text:
        return None
    last = None
    start = 0
    lower = text.lower()
    while True:
        idx = lower.find("boxed", start)
        if idx < 0:
            break
        tail = text[idx + len("boxed") :].lstrip()
        if tail.startswith("{"):
            last = _balanced_brace(tail[1:])
        else:
            m = _NUM_TOKEN.search(tail)
            last = m.group(0) if m else None
        start = idx + 5
    return last


def _balanced_brace(text: str) -> str | None:
    depth = 1
    out: list[str] = []
    for ch in text:
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
    return None


def normalize_numeric(value: str) -> str:
    s = (value or "").strip().replace(",", "").replace("$", "").replace("{", "").replace("}", "")
    s = s.strip()
    if not s:
        return ""
    try:
        number = float(s)
    except ValueError:
        return s
    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def normalize_mgsm_number(value: str) -> str:
    """Normalize a numeric token, including Vietnamese thousands/decimal marks."""
    s = (value or "").strip().replace("$", "").replace("{", "").replace("}", "")
    s = s.replace(" ", "")
    if not s:
        return ""
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if parts[0].lstrip("+-").isdigit() and all(p.isdigit() and len(p) == 3 for p in parts[1:]):
            s = "".join(parts)
        elif len(parts) == 2 and parts[0].lstrip("+-").isdigit() and parts[1].isdigit():
            s = parts[0] + "." + parts[1]
    elif "." in s:
        parts = s.split(".")
        if (
            len(parts) >= 2
            and parts[0].lstrip("+-").isdigit()
            and all(p.isdigit() and len(p) == 3 for p in parts[1:])
        ):
            s = "".join(parts)
    try:
        number = float(s)
    except ValueError:
        return s
    if number.is_integer():
        return str(int(number))
    return format(number, "g")


def numeric_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _NUM_TOKEN.finditer(text or "")]


def normalize_polymath(value: str) -> str:
    s = (value or "").strip()
    s = s.replace("$$", "").replace("$", "")
    s = re.sub(r"\s+", "", s)
    return s


def _after_answer_prefix(text: str) -> str | None:
    lower = text.lower()
    best = None
    best_pos = -1
    for prefix in _ANSWER_PREFIXES:
        pos = lower.rfind(prefix)
        if pos > best_pos:
            best_pos = pos
            best = text[pos + len(prefix) :].strip()
    return best


def _prefix_tail(text: str, prefix: str) -> str | None:
    if not text or not prefix:
        return None
    folded = strip_vietnamese_diacritics((text or "").lower())
    needle = strip_vietnamese_diacritics(prefix.lower()).rstrip(":")
    if not needle:
        return None
    pos = folded.rfind(needle)
    if pos < 0:
        return None
    return text[pos + len(needle) :].lstrip(" \t\r\n:*.")


def extract_gsm8k_gold(answer: str) -> str:
    """Numeric gold after the last GSM8K ``####`` delimiter."""
    text = str(answer or "")
    tail = text.rsplit("####", 1)[-1] if "####" in text else text
    toks = numeric_tokens(tail)
    if toks:
        return normalize_mgsm_number(toks[-1])
    return tail.strip()


def parse_mgsm_prefix(span: str, prefix: str | None = None) -> str:
    """Official Global-MGSM extractor: last number after the last answer prefix.

    Answer span only. Missing prefix → empty (no last-number fallback).
    """
    tail = _prefix_tail(span or "", prefix or MGSM_VI_ANSWER_PREFIX)
    if tail is None:
        return ""
    toks = numeric_tokens(tail)
    if not toks:
        return ""
    return normalize_mgsm_number(toks[-1])


def parse_numeric(span: str, raw: str = "") -> str:
    blob = span or ""
    boxed = extract_boxed(blob) or (extract_boxed(raw) if not blob else None)
    if boxed:
        return normalize_mgsm_number(boxed) or normalize_numeric(boxed)
    prefixed = _after_answer_prefix(blob)
    if prefixed:
        toks = numeric_tokens(prefixed)
        if toks:
            return normalize_mgsm_number(toks[-1])
    toks = numeric_tokens(blob)
    if toks:
        return normalize_mgsm_number(toks[-1])
    return ""


def _allowed_letters(options: Sequence[str | None] | None, alphabet: tuple[str, ...]) -> tuple[str, ...]:
    if not options:
        return alphabet
    present = []
    for letter, opt in zip(alphabet, options):
        if opt is None:
            continue
        if str(opt).strip() == "":
            continue
        present.append(letter)
    return tuple(present) or alphabet


def mcq_alphabet(task: TaskKind | str) -> tuple[str, ...] | None:
    if task == "mcq_aj":
        return MCQ_AJ
    if task == "mcq_ae":
        return MCQ_AE
    return None


_CHOICE_EDGE = ".*:;,_-()[]{} \t\n*"
_BOXED_LETTER = re.compile(r"^\\boxed\s*\{\s*([A-J])\s*\}\s*$", re.IGNORECASE)


def is_choice_only_span(span: str, alphabet: tuple[str, ...] | None) -> bool:
    """True when the span is a lone MCQ letter (C, (C), \\boxed{C}). Verbal wrappers still run LID."""
    if not alphabet:
        return False
    blob = (span or "").strip()
    if not blob:
        return False
    boxed = _BOXED_LETTER.match(blob)
    if boxed:
        return boxed.group(1).upper() in alphabet
    candidate = blob.strip(_CHOICE_EDGE)
    if len(candidate) != 1:
        return False
    return candidate.upper() in alphabet


def parse_mcq(
    span: str,
    raw: str = "",
    allowed: tuple[str, ...] = MCQ_AJ,
    options: Sequence[str | None] | None = None,
) -> str:
    """Parse a choice letter from the answer span.

    CoT is not searched for letters. The only exception is a *choice-only*
    generation (raw or unclosed think block is just ``A`` / ``(B)`` / ``\\boxed{C}``),
    which VMLU-style leaderboard prompts produce on purpose.
    """
    letters = _allowed_letters(options, allowed)
    blob = span or ""
    if not blob:
        if is_choice_only_span(raw, letters):
            blob = (raw or "").strip()
        else:
            inner = split_think_answer(raw)
            if not inner.answer_span and is_choice_only_span(inner.thinking, letters):
                blob = inner.thinking
    if not blob:
        return ""
    boxed = extract_boxed(blob)
    if boxed:
        letter = boxed.strip().upper()[:1]
        if letter in letters:
            return letter
    prefixed = _after_answer_prefix(blob)
    if prefixed:
        matches = [m.group(1).upper() for m in _LETTER.finditer(prefixed) if m.group(1).upper() in letters]
        return matches[0] if matches else ""
    matches = [m.group(1).upper() for m in _LETTER.finditer(blob) if m.group(1).upper() in letters]
    return matches[-1] if matches else ""


def parse_polymath(span: str, raw: str = "") -> str:
    blob = span or ""
    boxed = extract_boxed(blob) or (extract_boxed(raw) if not blob else None)
    if boxed:
        numeric = normalize_numeric(boxed)
        if numeric and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
            return numeric
        return normalize_polymath(boxed)
    return parse_numeric(span, raw if not blob else "")


def parse_answer(
    span: str,
    task: TaskKind | str,
    raw: str = "",
    options: Sequence[str | None] | None = None,
    extract: str | None = None,
) -> str:
    if extract == "mmlu_prox" or (extract and extract.startswith("mmlu_prox:")):
        return parse_mcq(span, raw, MCQ_AJ, options)
    if extract == "polymath":
        return parse_polymath(span, raw)
    if task == "numeric":
        return parse_numeric(span, raw)
    if task == "mcq_aj":
        return parse_mcq(span, raw, MCQ_AJ, options)
    if task == "mcq_ae":
        return parse_mcq(span, raw, MCQ_AE, options)
    if task == "polymath":
        return parse_polymath(span, raw)
    if task == "instruction":
        return (span or raw or "").strip()
    raise ValueError(f"unknown task {task!r}")


def answers_match(parsed: str, gold: str | None, task: TaskKind | str) -> bool:
    if not parsed or gold is None:
        return False
    if task == "numeric":
        return normalize_numeric(parsed) == normalize_numeric(str(gold))
    if task in {"mcq_aj", "mcq_ae"}:
        return parsed.strip().upper() == str(gold).strip().upper()[:1]
    if task == "polymath":
        return (
            normalize_polymath(parsed) == normalize_polymath(str(gold))
            or normalize_numeric(parsed) == normalize_numeric(str(gold))
        )
    return parsed.strip() == str(gold).strip()

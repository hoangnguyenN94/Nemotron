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
from collections.abc import Sequence

from nemotron.steps.eval.mr_l2.constants import COPY_MIN_CHARS, MCQ_AJ
from nemotron.steps.eval.mr_l2.textnorm import (
    collapse_ws,
    has_devanagari,
    has_vietnamese_diacritics,
    is_numeric_or_symbol,
    normalize_for_copy,
    tokenize_words,
)

# Maths, code, and common LaTeX environments — scored as allowed spans, not LID.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]+`")
_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)
_LATEX_PAREN = re.compile(r"\\\((?:.|\n)*?\\\)")
_LATEX_BRACK = re.compile(r"\\\[(?:.|\n)*?\\\]")
_LATEX_ENV = re.compile(
    r"\\begin\{(?:equation|align|gather|multline)\*?\}(?:.|\n)*?\\end\{(?:equation|align|gather|multline)\*?\}"
)
_QUOTE = re.compile(r"[“\"]([^”\"]{3,})[”\"]")

_SENT_SPLIT = re.compile(r"(?<=[\.!?…。！？\n])\s+")
_OPT_PREFIX = re.compile(rf"^[{''.join(MCQ_AJ)}][\.\)\:]\s+", re.I)
_OPT_LABEL = re.compile(rf"\b[{''.join(MCQ_AJ)}][\.\)\:]\s*", re.I)
_OPT_TRAIL = re.compile(rf"^[0-9.,%\s/+\-*×÷=()]+[{''.join(MCQ_AJ)}]\.?\s*$", re.I)
_MATH_OP = re.compile(r"[=+\-*/×÷^_\\()]")
_CODE_HEAD = re.compile(
    r"^(if|for|while|return|def|class|int|void|public|private|function|const|var|let)\b",
    re.I,
)
_CODE_TOKEN = re.compile(
    r"\b(if|for|while|return|def|class|int|void|null|true|false|"
    r"function|const|var|let|display|print)\b",
    re.I,
)


def split_sentences(text: str) -> list[str]:
    blob = (text or "").replace("\r\n", "\n")
    if not blob.strip():
        return []
    out: list[str] = []
    for line in blob.split("\n"):
        line = collapse_ws(line)
        if not line:
            continue
        out.extend(p.strip() for p in _SENT_SPLIT.split(line) if p.strip())
    return out


def strip_allowed_spans(text: str) -> str:
    s = text or ""
    for pat in (_FENCE, _LATEX_ENV, _DISPLAY, _LATEX_PAREN, _LATEX_BRACK, _INLINE_MATH, _INLINE_CODE):
        s = pat.sub(" ", s)
    s = _QUOTE.sub(" ", s)
    return collapse_ws(s)


def _copied_from_question(sentence: str, question_norm: str) -> bool:
    sent_norm = normalize_for_copy(sentence)
    if len(sent_norm) < COPY_MIN_CHARS:
        return False
    if sent_norm in question_norm:
        return True
    # Long overlapping token window: 8+ words copied in order.
    q_words = question_norm.split()
    s_words = sent_norm.split()
    if len(s_words) < 8 or len(q_words) < 8:
        return False
    q_join = " ".join(q_words)
    for i in range(0, len(s_words) - 7):
        window = " ".join(s_words[i : i + 8])
        if window in q_join:
            return True
    return False


def strip_sentence(sentence: str, question: str = "") -> str:
    remainder = strip_allowed_spans(sentence)
    if question and _copied_from_question(remainder, normalize_for_copy(question)):
        return ""
    return remainder


def _is_identifier_formula(text: str) -> bool:
    """True for leftover identifier algebra such as ``S = (a+b)*h/2``."""
    if not _MATH_OP.search(text or ""):
        return False
    if has_vietnamese_diacritics(text) or has_devanagari(text):
        return False
    words = tokenize_words(text)
    if not words:
        return True
    return all(len(w) <= 2 or w.isdigit() for w in words)


def is_option_recap(text: str) -> bool:
    """MCQ option restatement: ``A. 3,125`` / ``3,125 B.`` / ``A. 3,125 B. 31250/10000``."""
    s = collapse_ws(text)
    if not s:
        return True
    if _OPT_TRAIL.match(s):
        return True
    if _OPT_PREFIX.match(s) or _OPT_LABEL.search(s):
        rest = _OPT_LABEL.sub(" ", s)
        return is_numeric_or_symbol(rest) or _is_identifier_formula(rest)
    return False


def is_code_remainder(text: str) -> bool:
    """Bare code / pseudocode that is not wrapped in fences.

    A Vietnamese (or other native) carrier sentence that merely mentions a
    keyword like ``int`` is still scored; only code-shaped remainders are
    allowed.
    """
    s = collapse_ws(text)
    if not s:
        return True
    if has_vietnamese_diacritics(s) or has_devanagari(s):
        return False
    if _CODE_HEAD.match(s):
        return True
    if _CODE_TOKEN.search(s) and any(ch in s for ch in "{};"):
        return True
    return False


def is_allowed_remainder(text: str) -> bool:
    """Remainders that carry no native-language claim after strip."""
    s = collapse_ws(text)
    if not s or is_numeric_or_symbol(s):
        return True
    if is_option_recap(s) or _is_identifier_formula(s) or is_code_remainder(s):
        return True
    return False


def copy_corpus(question: str = "", options: Sequence[str | None] | None = None) -> str:
    parts = [question or ""]
    for opt in options or ():
        if opt and str(opt).strip():
            parts.append(str(opt))
    return "\n".join(p for p in parts if p)


def strip_thinking(
    thinking: str,
    question: str = "",
    options: Sequence[str | None] | None = None,
) -> list[tuple[str, str]]:
    """Return (original_sentence, remainder_after_strip) pairs."""
    q_norm = normalize_for_copy(copy_corpus(question, options))
    out: list[tuple[str, str]] = []
    for sent in split_sentences(thinking):
        remainder = strip_allowed_spans(sent)
        if q_norm and remainder and _copied_from_question(remainder, q_norm):
            remainder = ""
        elif is_allowed_remainder(remainder):
            remainder = ""
        out.append((sent, remainder))
    return out

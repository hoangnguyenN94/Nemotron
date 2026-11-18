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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from nemotron.steps.eval.mr_l2.constants import IN_LANGUAGE_THRESHOLD
from nemotron.steps.eval.mr_l2.parser import is_choice_only_span, mcq_alphabet
from nemotron.steps.eval.mr_l2.strip import is_allowed_remainder, strip_thinking
from nemotron.steps.eval.mr_l2.textnorm import (
    collapse_ws,
    has_devanagari,
    has_vietnamese_diacritics,
    is_numeric_or_symbol,
    strip_vietnamese_diacritics,
    tokenize_words,
)
from nemotron.steps.eval.mr_l2.types import LanguageScore, TaskKind

# Closed function-word lists. Short tokens are included only where they are
# distinctive inside a *set* score, not used alone.
_VI_FUNCTION = {
    "khong",
    "duoc",
    "nguoi",
    "nhung",
    "trong",
    "mot",
    "cua",
    "cho",
    "voi",
    "nay",
    "viec",
    "nhu",
    "thi",
    "dang",
    "cung",
    "hay",
    "neu",
    "khi",
    "tai",
    "tren",
    "duoi",
    "sau",
    "truoc",
    "lam",
    "den",
    "roi",
    "nua",
    "rat",
    "bang",
    "hoac",
    "giua",
    "ngoai",
    "theo",
    "qua",
    "van",
    "biet",
    "giai",
    "toan",
    "tinh",
    "bang",
    "phai",
    "dung",
    "sai",
    "vay",
    "the",
    "de",
    "tu",
    "ra",
    "vao",
    "lai",
    "con",
    "cai",
    "cac",
    "nhieu",
    "it",
    "hon",
    "nhat",
    "moi",
    "da",
    "se",
    "la",
    "va",
    "bi",
    "do",
    "vi",
    "tu",
}

_EN_FUNCTION = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "is",
    "that",
    "for",
    "with",
    "as",
    "on",
    "at",
    "this",
    "from",
    "by",
    "or",
    "an",
    "be",
    "are",
    "was",
    "were",
    "been",
    "have",
    "has",
    "had",
    "not",
    "but",
    "if",
    "we",
    "you",
    "they",
    "it",
    "will",
    "would",
    "can",
    "should",
    "then",
    "than",
    "into",
    "its",
    "our",
    "their",
    "there",
    "which",
    "because",
    "solve",
    "step",
    "answer",
    "therefore",
    "thus",
    "hence",
}


class LanguageDetector(Protocol):
    """Return an ISO 639-1 code, or None if the sentence should not be scored."""

    def detect(self, text: str) -> str | None: ...


@dataclass(frozen=True)
class HeuristicDetector:
    """
    Offline detector for Vietnamese (accented *and* unaccented) vs English.

    FastText is optional and wrapped separately. Tests and smoke runs use this
    backend so the contract does not depend on downloading lid.176.bin.
    """

    def detect(self, text: str) -> str | None:
        blob = (text or "").strip()
        if not blob or is_numeric_or_symbol(blob):
            return None
        if has_devanagari(blob):
            return "hi"
        if has_vietnamese_diacritics(blob):
            return "vi"
        words = [strip_vietnamese_diacritics(w).lower() for w in tokenize_words(blob)]
        if not words:
            return None
        vi = sum(1 for w in words if w in _VI_FUNCTION)
        en = sum(1 for w in words if w in _EN_FUNCTION)
        if vi == 0 and en == 0:
            return None
        if vi > en:
            return "vi"
        if en > vi:
            return "en"
        # Tie: unaccented Vietnamese often shares short tokens with English.
        # Prefer vi when any distinctive VN token of length >= 4 is present.
        if any(len(w) >= 4 and w in _VI_FUNCTION for w in words):
            return "vi"
        return "en"


class FastTextDetector:
    """Optional FastText backend. Falls back to heuristic if the model is absent."""

    def __init__(self, model_path: str, fallback: LanguageDetector | None = None) -> None:
        self._fallback = fallback or HeuristicDetector()
        self._model = None
        try:
            import fasttext  # type: ignore

            self._model = fasttext.load_model(model_path)
        except Exception:
            self._model = None

    def detect(self, text: str) -> str | None:
        blob = (text or "").strip()
        if not blob or is_numeric_or_symbol(blob):
            return None
        # Accent-aware Vietnamese must win even if FastText says 'en'
        # on unaccented text — that is the Nemotron contract.
        heuristic = self._fallback.detect(blob)
        if heuristic in {"vi", "hi"}:
            return heuristic
        if self._model is None:
            return heuristic
        label = self._model.predict(blob.replace("\n", " "), k=1)[0][0]
        code = label.replace("__label__", "")
        if code == "vie":
            return "vi"
        if len(code) >= 2:
            return code[:2]
        return heuristic


def default_detector() -> LanguageDetector:
    return HeuristicDetector()


def _word_weight(text: str) -> int:
    return max(1, len(tokenize_words(text)))


def _substantial_latin_prose(text: str) -> bool:
    """True when the remainder is English-like prose with no LID lexicon hit."""
    words = [w for w in tokenize_words(text) if w.isascii() and w.isalpha() and len(w) >= 3]
    return len(words) >= 3


def score_thinking_language(
    thinking: str,
    target: str,
    question: str = "",
    detector: LanguageDetector | None = None,
    threshold: float = IN_LANGUAGE_THRESHOLD,
    options: Sequence[str | None] | None = None,
) -> LanguageScore:
    """
    Word-weighted LID after strip.

    Scored remainders are those with a language signal. Option recap, identifier
    formulas, bare code, numeric-only spans, copied question/option text, and
    LID-none fragments with no Latin prose are allowed and do not enter the
    denominator. Share is matching-word-mass / scored-word-mass. If nothing is
    scored, share is 1.0.
    """
    det = detector or default_detector()
    langs: list[str | None] = []
    n_scored = 0
    n_allowed = 0
    w_scored = 0
    w_match = 0
    pairs = strip_thinking(thinking, question, options)
    if not pairs:
        return LanguageScore(1.0, True, 0, 0, 0, ())

    for _, remainder in pairs:
        rem = (remainder or "").strip()
        if not rem or is_numeric_or_symbol(rem) or is_allowed_remainder(rem):
            n_allowed += 1
            langs.append(None)
            continue
        lang = det.detect(rem)
        if lang is None and not _substantial_latin_prose(rem):
            n_allowed += 1
            langs.append(None)
            continue
        n_scored += 1
        weight = _word_weight(rem)
        w_scored += weight
        if lang is None:
            lang = "und"
        langs.append(lang)
        if lang == target:
            w_match += weight

    share = 1.0 if w_scored == 0 else w_match / w_scored
    return LanguageScore(
        share=share,
        in_language_95=share >= threshold,
        n_sentences=len(pairs),
        n_scored=n_scored,
        n_allowed=n_allowed,
        languages=tuple(langs),
    )


def detect_answer_language(
    span: str,
    requested: str,
    detector: LanguageDetector | None = None,
    *,
    task: TaskKind | str | None = None,
) -> tuple[str, bool]:
    """Return (answer_language, language_ok). Numeric and lone MCQ letters are exempt."""
    blob = collapse_ws(span or "")
    if not blob:
        return "empty", False
    if is_choice_only_span(blob, mcq_alphabet(task or "")):
        return "choice", True
    if is_numeric_or_symbol(blob):
        return "numeric", True
    det = detector or default_detector()
    lang = det.detect(blob) or "unknown"
    return lang, lang == requested

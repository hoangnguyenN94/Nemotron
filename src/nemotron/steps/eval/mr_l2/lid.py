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
    has_distinctive_vietnamese_diacritics,
    has_vietnamese_diacritics,
    is_numeric_or_symbol,
    strip_vietnamese_diacritics,
    tokenize_words,
)
from nemotron.steps.eval.mr_l2.types import LanguageScore, TaskKind

# Minimum alphabetic characters for a remainder to count as a language claim
# when the detector returns None. Script-agnostic; not a per-language rule.
_LETTER_MASS = 4

# ISO 639-2 / FastText extras → ISO 639-1. Not a language policy.
_ISO3_TO_1 = {
    "vie": "vi",
    "eng": "en",
    "hin": "hi",
    "fra": "fr",
    "spa": "es",
    "deu": "de",
    "rus": "ru",
    "zho": "zh",
    "cmn": "zh",
    "jpn": "ja",
    "kor": "ko",
    "tha": "th",
    "ara": "ar",
}


class LanguageDetector(Protocol):
    """Return an ISO 639-1 code, or None if this backend has no signal."""

    def detect(self, text: str) -> str | None: ...


@dataclass(frozen=True)
class CascadingDetector:
    """First non-None backend wins. Swap/reorder freely; the scorer does not care."""

    detectors: tuple[LanguageDetector, ...]

    def detect(self, text: str) -> str | None:
        blob = (text or "").strip()
        if not blob or is_numeric_or_symbol(blob):
            return None
        for det in self.detectors:
            lang = det.detect(blob)
            if lang:
                return lang
        return None


@dataclass(frozen=True)
class ScriptDetector:
    """Map a dominant *non-Latin* Unicode script to an ISO code.

    Latin is left to a lexical/model backend (script is not a language).
    The label only needs to be comparable to the L2 target: Han vs vi is a miss
    even if Han is labeled zh rather than ja.
    """

    min_letters: int = _LETTER_MASS

    def detect(self, text: str) -> str | None:
        counts: dict[str, int] = {}
        for ch in text or "":
            if not ch.isalpha():
                continue
            lang = _lang_for_char(ch)
            if not lang:
                continue
            counts[lang] = counts.get(lang, 0) + 1
        if not counts:
            return None
        lang, n = max(counts.items(), key=lambda item: item[1])
        return lang if n >= self.min_letters else None


def _lang_for_char(ch: str) -> str | None:
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return "hi"
    if 0x0980 <= cp <= 0x09FF:
        return "bn"
    if 0x0E00 <= cp <= 0x0E7F:
        return "th"
    if 0x0400 <= cp <= 0x052F or 0x2DE0 <= cp <= 0x2DFF or 0xA640 <= cp <= 0xA69F:
        return "ru"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "ar"
    if 0x0590 <= cp <= 0x05FF:
        return "he"
    if 0x0370 <= cp <= 0x03FF:
        return "el"
    if 0xAC00 <= cp <= 0xD7AF:
        return "ko"
    if 0x3040 <= cp <= 0x30FF:
        return "ja"
    if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
        return "zh"
    return None


# --- Latin adapter (offline). Not consulted by the scoring formula itself. ---

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
}
_VI_STRONG = {w for w in _VI_FUNCTION if len(w) >= 4}
_VI_WEAK = {w for w in _VI_FUNCTION if len(w) < 4}

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


@dataclass(frozen=True)
class HeuristicDetector:
    """Offline Latin adapter: Vietnamese (accented and không-dấu) vs English.

    Used when FastText/GlotLID is not loaded. ``None`` means no signal, not a
    language id — the scorer does not invent ``und``. Not a substitute for a
    model LID on French/Spanish/Chinese.
    """

    def detect(self, text: str) -> str | None:
        blob = (text or "").strip()
        if not blob or is_numeric_or_symbol(blob):
            return None
        vi_strong = 0
        vi_weak = 0
        en = 0
        for raw in tokenize_words(blob):
            folded = strip_vietnamese_diacritics(raw).lower()
            if has_distinctive_vietnamese_diacritics(raw):
                vi_strong += 1
                continue
            if folded in _VI_STRONG:
                vi_strong += 1
                continue
            if folded in _VI_WEAK and has_vietnamese_diacritics(raw):
                vi_strong += 1
                continue
            if folded in _EN_FUNCTION:
                en += 1
                continue
            if folded in _VI_WEAK:
                vi_weak += 1
        vi = vi_strong + (vi_weak if vi_strong else 0)
        if vi == 0 and en == 0:
            return None
        if vi > en:
            return "vi"
        if en > vi:
            return "en"
        return "vi" if vi_strong else "en"


class FastTextDetector:
    """Optional FastText/GlotLID-style backend. No per-language overrides."""

    def __init__(self, model_path: str, fallback: LanguageDetector | None = None) -> None:
        self._fallback = fallback
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
        if self._model is None:
            return self._fallback.detect(blob) if self._fallback else None
        label = self._model.predict(blob.replace("\n", " "), k=1)[0][0]
        code = str(label).replace("__label__", "").replace("-Latn", "").replace("_Latn", "")
        mapped = _ISO3_TO_1.get(code, code)
        if len(mapped) >= 2:
            return mapped[:2]
        return self._fallback.detect(blob) if self._fallback else None


def default_detector() -> LanguageDetector:
    """Script (non-Latin) then Vietnamese/English Latin adapter. Plug in FastText via Cascade."""
    return CascadingDetector((ScriptDetector(), HeuristicDetector()))


def _word_weight(text: str) -> int:
    return max(1, len(tokenize_words(text)))


def _has_letter_mass(text: str, min_letters: int = _LETTER_MASS) -> bool:
    n = 0
    for ch in text or "":
        if ch.isalpha():
            n += 1
            if n >= min_letters:
                return True
    return False


# Diagnostic scripts still import this name.
_substantial_latin_prose = _has_letter_mass


def thinking_lang_ok_for_gate(score: LanguageScore) -> bool:
    """Hard-gate thinking check.

    Language identity is entirely in ``in_language_95`` (detector vs target).
    Empty thinking (no sentences) fails. Language-neutral remainders
    (math/code/options) leave n_scored=0 but n_sentences>0 and do not fail.
    Unidentified letter-mass with no ISO code is not a pass: the scorer sets
    share=0 when nothing was identified.
    """
    if not score.in_language_95:
        return False
    if score.n_scored >= 1:
        return True
    return score.n_sentences > 0


def score_thinking_language(
    thinking: str,
    target: str,
    question: str = "",
    detector: LanguageDetector | None = None,
    threshold: float = IN_LANGUAGE_THRESHOLD,
    options: Sequence[str | None] | None = None,
) -> LanguageScore:
    """
    Word-weighted LID after structural strip. Language-agnostic.

    1. Strip math, code, option recap, copied prompt text.
    2. Ask ``detector`` for an ISO code (any backend).
    3. Detector None + no letter mass → allowed, out of the denominator.
    4. Detector None + letter mass → unidentified (no ISO claim). Out of the
       denominator when *any* remainder was identified; if nothing was
       identified, share=0 (cannot confirm the target). The scorer never
       invents a language code.
    5. share = matching-word-mass / scored-word-mass; empty scored mass and
       no unidentified prose → 1.0.
    """
    det = detector or default_detector()
    langs: list[str | None] = []
    n_scored = 0
    n_allowed = 0
    n_unidentified = 0
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
        if lang is None:
            if _has_letter_mass(rem):
                n_unidentified += 1
                langs.append(None)
                continue
            n_allowed += 1
            langs.append(None)
            continue
        n_scored += 1
        weight = _word_weight(rem)
        w_scored += weight
        langs.append(lang)
        if lang == target:
            w_match += weight

    if w_scored == 0:
        share = 0.0 if n_unidentified else 1.0
    else:
        share = w_match / w_scored
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

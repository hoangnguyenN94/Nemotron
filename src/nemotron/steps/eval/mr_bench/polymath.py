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

"""Official PolyMath prompt, boxed extractor, math_equal, and DW-ACC.

Matches QwenLM/PolyMath ``instruction.py`` (Vietnamese ``query_dic``),
``eval/run_eval.py`` (first ``boxed{…}`` on the answer span after deleting ASCII
spaces, then ``math_equal``), and the paper DW-ACC weights 1/2/4/8.

Hindi (``polymath_hi``) is TUM-NLP/PluraMath, which extends PolyMath to Hindi
with the same 500-item / four-level layout. Qwen PolyMath has no ``hi`` split;
the boxed note is PluraMath ``instructions_prompts.INSTRUCTION["hi"]``.
"""

from __future__ import annotations

import re
from math import isclose
from typing import Any, Mapping

# QwenLM/PolyMath instruction.py query_dic["vi"] after Python parses the
# ``$\\boxed\{\}$`` literal: empty \boxed{} in the note, not a filled answer.
POLYMATH_VI_INSTRUCTION = r"Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\boxed{}$."
# TUM-NLP/PluraMath instructions_prompts.py INSTRUCTION["hi"].
POLYMATH_HI_INSTRUCTION = r"नोट: कृपया अंतिम उत्तर को $\boxed{}$ में रखें।"
# QwenLM/PolyMath instruction.py query_dic["en"].
POLYMATH_EN_INSTRUCTION = r"Note: Please put the final answer in the $\boxed{}$."
POLYMATH_INSTRUCTIONS = {
    "vi": POLYMATH_VI_INSTRUCTION,
    "hi": POLYMATH_HI_INSTRUCTION,
    "en": POLYMATH_EN_INSTRUCTION,
}
POLYMATH_SOURCES = frozenset({"polymath_vi", "polymath_hi", "polymath_en"})

# Paper §3 / App. C.4 reasoning-model decoding (pass@1 here; they report average@16).
POLYMATH_TEMPERATURE = 0.6
POLYMATH_TOP_P = 0.95
POLYMATH_TOP_K = 20
POLYMATH_MAX_TOKENS = 65_536

POLYMATH_LEVELS = ("low", "medium", "high", "top")
POLYMATH_WEIGHTS = (1, 2, 4, 8)

_BOXED_OPEN = re.compile(r"boxed\{")
_FRAC = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_SIMPLE_FRAC = re.compile(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$")


def is_polymath_source(name: str | None) -> bool:
    return (name or "") in POLYMATH_SOURCES


def is_polymath_benchmark(name: str | None) -> bool:
    return is_polymath_source(name)


def polymath_lang(name: str | None) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key.endswith("_hi") or key == "hi":
        return "hi"
    if key.endswith("_en") or key == "en":
        return "en"
    return "vi"


def polymath_instruction(lang: str = "vi") -> str:
    try:
        return POLYMATH_INSTRUCTIONS[lang]
    except KeyError as exc:
        raise ValueError(f"no PolyMath boxed note for {lang!r}") from exc


def is_polymath_official(prompt_style: str | None, source: str | None = None) -> bool:
    if prompt_style != "polymath":
        return False
    if source is None:
        return True
    return is_polymath_source(source)


def format_polymath_prompt(
    question: str,
    instruction: str | None = None,
    *,
    lang: str = "vi",
) -> str:
    """Official concat: question + one newline + language-specific boxed note.

    evalscope / Qwen inference appends the note with ``\\n``, not the MGSM
    ``instruction + "\\n\\n" + question`` layout. Trailing question whitespace
    is kept.
    """
    note = instruction if instruction is not None else polymath_instruction(lang)
    return f"{question or ''}\n{note}"


def extract_first_boxed_nospace(text: str) -> str | None:
    """Qwen ``run_eval.extract_boxed_content``: delete ASCII spaces, take first boxed."""
    compact = (text or "").replace(" ", "")
    results: list[str] = []
    for match in _BOXED_OPEN.finditer(compact):
        start = match.end()
        depth = 1
        i = start
        while i < len(compact) and depth > 0:
            if compact[i] == "{":
                depth += 1
            elif compact[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(compact[start : i - 1])
    return results[0] if results else None


def parse_polymath_official(span: str) -> str:
    """Answer-span only. No thinking search, no last-number fallback."""
    boxed = extract_first_boxed_nospace(span or "")
    return boxed if boxed is not None else ""


def _strip_math_wrap(value: str) -> str:
    s = (value or "").strip()
    s = s.replace("$$", "").replace("$", "")
    s = s.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\s+", "", s)
    return s


def _parse_digits(num: str) -> float | None:
    blob = regex_sub_commas(str(num))
    try:
        return float(blob)
    except ValueError:
        if blob.endswith("%"):
            blob = blob[:-1].rstrip("\\")
            try:
                return float(blob) / 100
            except ValueError:
                return None
        return None


def regex_sub_commas(num: str) -> str:
    return re.sub(r",", "", str(num))


def _try_number(value: str) -> float | None:
    s = _strip_math_wrap(value)
    if not s:
        return None
    parsed = _parse_digits(s)
    if parsed is not None:
        return parsed
    frac = _FRAC.fullmatch(s) or _FRAC.search(s)
    if frac and s == frac.group(0):
        num, den = _try_number(frac.group(1)), _try_number(frac.group(2))
        if num is not None and den not in (None, 0.0):
            return num / den
    slash = _SIMPLE_FRAC.fullmatch(s)
    if slash:
        num, den = _try_number(slash.group(1)), _try_number(slash.group(2))
        if num is not None and den not in (None, 0.0):
            return num / den
    return None


def _numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _symbolic_equal(pred: str, gold: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False

    def _parse(raw: str):
        s = _strip_math_wrap(raw)
        s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
        s = s.replace(r"\pi", "pi").replace(r"\infty", "oo")
        s = _FRAC.sub(r"((\1)/(\2))", s)
        s = s.replace("{", "(").replace("}", ")")
        s = s.replace("^", "**")
        s = s.replace("\\", "")
        try:
            return parse_expr(s, evaluate=True)
        except Exception:
            return None

    a, b = _parse(pred), _parse(gold)
    if a is None or b is None:
        return False
    try:
        return bool(simplify(a - b) == 0)
    except Exception:
        return False


def math_equal(prediction: str | None, reference: str | None) -> bool:
    """Qwen ``scripts.math_equal`` core: string, numeric (±%), then symbolic."""
    if prediction is None or reference is None:
        return False
    pred = str(prediction).strip()
    gold = str(reference).strip()
    if not pred:
        return False
    if pred.lower() == gold.lower():
        return True
    if _strip_math_wrap(pred).lower() == _strip_math_wrap(gold).lower() and _strip_math_wrap(pred):
        return True

    pred_n, gold_n = _try_number(pred), _try_number(gold)
    if pred_n is not None and gold_n is not None:
        if _numeric_equal(pred_n, gold_n):
            return True
        for item in (gold_n / 100, gold_n, gold_n * 100):
            if _numeric_equal(pred_n, item):
                return True
        return False

    return _symbolic_equal(pred, gold)


def difficulty_weighted_acc(by_stratum: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """DW-ACC = (1 a_low + 2 a_med + 4 a_high + 8 a_top) / 15.

    ``dw_acc`` is the exact fraction. ``official_score`` is Qwen ``score.json``:
    per-level accuracy rounded to 0.1%, then weighted and rounded to 0.1.
    """
    accs: list[float] = []
    percents: list[float] = []
    missing: list[str] = []
    for level in POLYMATH_LEVELS:
        bucket = by_stratum.get(level) or {}
        scored = int(bucket.get("scored") or 0)
        if scored <= 0:
            missing.append(level)
            continue
        acc = float(bucket["correct"]) / scored
        accs.append(acc)
        percents.append(round(acc * 100, 1))
    if missing or len(accs) != 4:
        return {
            "dw_acc": None,
            "official_score": None,
            "missing_levels": missing,
            "level_acc_percent": None,
        }
    dw = sum(w * a for w, a in zip(POLYMATH_WEIGHTS, accs)) / 15
    official = round(sum(w * p for w, p in zip(POLYMATH_WEIGHTS, percents)) / 15, 1)
    return {
        "dw_acc": dw,
        "official_score": official,
        "missing_levels": [],
        "level_acc_percent": dict(zip(POLYMATH_LEVELS, percents)),
    }

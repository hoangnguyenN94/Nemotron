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

"""Classify L2 remainders and summarise fail modes.

L2 is `in_language_95`: word-weighted share of scored thinking remainders in
the target language after strip, threshold 0.95. This module explains *why*
an item failed that rule (English closer, English CoT, mixed share, loop).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from nemotron.steps.eval.mr_l2.lid import _has_letter_mass, _word_weight, default_detector
from nemotron.steps.eval.mr_l2.strip import (
    _copied_from_question,
    _is_identifier_formula,
    is_code_remainder,
    is_option_recap,
    split_sentences,
    strip_allowed_spans,
)
from nemotron.steps.eval.mr_l2.textnorm import collapse_ws, is_numeric_or_symbol, normalize_for_copy

_DET = default_detector()

REASON_ORDER = (
    "option_recap",
    "formula",
    "code",
    "copied",
    "none_allowed",
    "numeric",
    "en",
    "und_latin",
    "other_lang",
    "vi",
)


def share_bucket(share: float | None) -> str:
    if share is None:
        return "na"
    if share < 0.5:
        return "<0.50"
    if share < 0.8:
        return "0.50-0.80"
    if share < 0.9:
        return "0.80-0.90"
    if share < 0.95:
        return "0.90-0.95"
    return ">=0.95"


def classify_remainder(remainder: str, question_norm: str, *, target: str = "vi") -> str:
    """Label one stripped remainder: allowed span, target match, or leak kind."""
    text = (remainder or "").strip()
    if not text or is_numeric_or_symbol(text):
        return "numeric"
    if _is_identifier_formula(text):
        return "formula"
    if is_code_remainder(text):
        return "code"
    if is_option_recap(text):
        return "option_recap"
    copied = bool(question_norm and _copied_from_question(text, question_norm))
    lang = _DET.detect(text)
    if lang is None:
        kind = "und_latin" if _has_letter_mass(text) else "none_allowed"
        return "copied" if copied else kind
    if lang == target:
        return "copied" if copied else "vi"
    if copied:
        return "copied"
    if lang == "en":
        return "en"
    return "other_lang"


def is_allowed_kind(kind: str) -> bool:
    return kind in {"numeric", "copied", "option_recap", "formula", "code", "none_allowed"}


def classify_thinking(
    thinking: str,
    question: str = "",
    *,
    target: str = "vi",
    options: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Per-sentence remainder kinds plus a primary fail/pass explanation."""
    parts = [question or ""]
    for option in options or ():
        if option and str(option).strip():
            parts.append(str(option))
    question_norm = normalize_for_copy("\n".join(p for p in parts if p))
    sentences = split_sentences(thinking or "")
    remainders: list[dict[str, Any]] = []
    scored_kinds: Counter[str] = Counter()
    allowed_kinds: Counter[str] = Counter()
    weight_scored = 0
    weight_match = 0
    for sentence in sentences:
        remainder = strip_allowed_spans(sentence)
        kind = classify_remainder(remainder, question_norm, target=target)
        weight = _word_weight(remainder) if remainder.strip() else 0
        pack = {
            "kind": kind,
            "weight": weight,
            "text": collapse_ws(sentence)[:240],
        }
        remainders.append(pack)
        if is_allowed_kind(kind):
            allowed_kinds[kind] += 1
            continue
        scored_kinds[kind] += 1
        weight_scored += max(1, weight)
        if kind == "vi":
            weight_match += max(1, weight)

    share = 1.0 if weight_scored == 0 else weight_match / weight_scored
    leaks = [row for row in remainders if row["kind"] in {"en", "und_latin", "other_lang"}]
    primary = _primary_reason(share, leaks, scored_kinds)
    return {
        "n_sentences": len(sentences),
        "n_scored": sum(scored_kinds.values()),
        "n_allowed": sum(allowed_kinds.values()),
        "share_approx": share,
        "share_bucket": share_bucket(share),
        "scored_kinds": dict(scored_kinds),
        "allowed_kinds": dict(allowed_kinds),
        "primary": primary,
        "leaks": leaks[:6],
        "remainders": remainders[:12],
    }


def _primary_reason(share: float, leaks: list[dict[str, Any]], scored_kinds: Counter[str]) -> str:
    if share >= 0.95:
        return "pass"
    if scored_kinds.get("en", 0) and share == 0.0:
        return "english_cot"
    if leaks and share >= 0.80:
        last = leaks[-1]["kind"]
        if last == "en":
            return "english_closer"
        return "mixed_share"
    if scored_kinds.get("und_latin"):
        return "und_latin"
    if scored_kinds.get("other_lang"):
        return "other_lang"
    if 0.5 <= share < 0.95:
        return "mixed_share"
    if scored_kinds.get("en"):
        return "english_cot"
    return "other"


def classify_record(record: Mapping[str, Any], *, target: str = "vi") -> dict[str, Any]:
    thinking = str(record.get("thinking") or "")
    question = str(record.get("question") or record.get("user_message") or "")
    options = record.get("options")
    if options is not None and not isinstance(options, (list, tuple)):
        options = None
    info = classify_thinking(thinking, question, target=target, options=options)
    if record.get("loop"):
        info["primary_with_closure"] = "loop"
    elif record.get("truncated"):
        info["primary_with_closure"] = "truncated"
    else:
        info["primary_with_closure"] = info["primary"]
    return info


def summarize_l2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate L2 rate, share buckets, and fail-mode counts."""
    n = len(rows)
    n_l2 = sum(1 for row in rows if row.get("in_language_95"))
    n_closed = sum(1 for row in rows if row.get("closed"))
    n_loop = sum(1 for row in rows if row.get("loop"))
    n_trunc = sum(1 for row in rows if row.get("truncated"))
    shares = [float(row["thinking_lang_share"]) for row in rows if isinstance(row.get("thinking_lang_share"), (int, float))]
    fail_primary: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    source_l2: dict[str, list[int]] = {}
    fail_samples: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source") or "unknown")
        source_l2.setdefault(source, [0, 0])
        source_l2[source][1] += 1
        if row.get("in_language_95"):
            source_l2[source][0] += 1
        buckets[share_bucket(row.get("thinking_lang_share") if isinstance(row.get("thinking_lang_share"), (int, float)) else None)] += 1
        if row.get("in_language_95"):
            continue
        primary = str(row.get("l2_primary") or "other")
        fail_primary[primary] += 1
        if len(fail_samples) < 24:
            fail_samples.append(
                {
                    "item_id": row.get("item_id"),
                    "source": source,
                    "thinking_lang_share": row.get("thinking_lang_share"),
                    "l2_primary": primary,
                    "closed": row.get("closed"),
                    "loop": row.get("loop"),
                    "leaks": row.get("l2_leaks") or [],
                }
            )
    by_source = {
        name: {"n": tot, "n_l2": ok, "in_language_95_rate": (ok / tot if tot else None)}
        for name, (ok, tot) in sorted(source_l2.items())
    }
    return {
        "n": n,
        "n_in_language_95": n_l2,
        "in_language_95_rate": (n_l2 / n if n else None),
        "n_closed": n_closed,
        "closed_rate": (n_closed / n if n else None),
        "n_loop": n_loop,
        "n_truncated": n_trunc,
        "mean_thinking_lang_share": (sum(shares) / len(shares) if shares else None),
        "share_buckets": dict(buckets),
        "fail_primary": dict(fail_primary),
        "by_source": by_source,
        "fail_samples": fail_samples,
        "formula": "word-weighted share of scored remainders after strip; pass if share >= 0.95",
    }

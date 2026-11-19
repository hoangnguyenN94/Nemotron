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
import re

from nemotron.steps.eval.mr_l2.textnorm import collapse_ws, nfc

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def stable_hash(text: str, n: int = 16) -> str:
    blob = collapse_ws(nfc(text)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]


def make_item_id(source: str, native_id: str) -> str:
    if not source or ":" in source:
        raise ValueError(f"invalid source {source!r}")
    native = str(native_id).strip()
    if not native:
        raise ValueError("empty native id")
    return f"{source}:{native}"


def mgsm_id(question: str, *, lang: str = "vi") -> str:
    return make_item_id(f"mgsm_{lang}", stable_hash(question))


def gsm8k_id(question: str, *, lang: str = "hi") -> str:
    return make_item_id(f"gsm8k_{lang}", stable_hash(question))


def gsm8k_hi_id(question: str) -> str:
    return gsm8k_id(question, lang="hi")


def gsm8k_train_vi_id(english_question: str) -> str:
    """Stable train ID: hash the English GSM8K source, not the Vietnamese text."""
    return make_item_id("gsm8k_train_vi", stable_hash(english_question))


def m_arena_hard_id(question_id: str, *, lang: str = "vi") -> str:
    cleaned = _SAFE.sub("-", str(question_id).strip())
    return make_item_id(f"m_arena_hard_{lang}", cleaned)


def mmlu_id(question_id: int | str, *, validation: bool = False, lang: str = "vi") -> str:
    source = f"mmlu_prox_{lang}_val" if validation else f"mmlu_prox_{lang}"
    return make_item_id(source, str(int(question_id)))


def polymath_id(native_id: str, *, lang: str = "vi") -> str:
    cleaned = _SAFE.sub("-", native_id.strip())
    return make_item_id(f"polymath_{lang}", cleaned)


def vmlu_id(native_id: str) -> str:
    cleaned = _SAFE.sub("-", native_id.strip())
    return make_item_id("vmlu", cleaned)


def milu_id(question: str, options: tuple[str, ...]) -> str:
    blob = "\n".join((question,) + options)
    return make_item_id("milu_1_5", stable_hash(blob))


def ifeval_id(key: int | str) -> str:
    return make_item_id("ifeval", str(int(key)))


def parse_item_id(item_id: str) -> tuple[str, str]:
    source, sep, native = item_id.partition(":")
    if not sep or not source or not native:
        raise ValueError(f"malformed item_id {item_id!r}")
    return source, native

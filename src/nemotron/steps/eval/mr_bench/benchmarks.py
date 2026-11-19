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

from dataclasses import dataclass
from typing import Literal

from nemotron.steps.eval.mr_l2.types import TaskKind

Backend = Literal["native", "nemo-evaluator"]

SCORED_BENCHMARKS = ("mgsm_vi", "mmlu_prox_vi", "polymath_vi")
DEFAULT_RUN_BENCHMARKS = SCORED_BENCHMARKS


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    sources: tuple[str, ...]
    task: TaskKind
    default_splits: tuple[str, ...] | None
    submission_csv: bool
    score_gold: bool
    description: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "mgsm_vi": BenchmarkSpec(
        name="mgsm_vi",
        sources=("mgsm_vi",),
        task="numeric",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="Global-MGSM Vietnamese (250 grade-school numeric items)",
    ),
    "mgsm_en": BenchmarkSpec(
        name="mgsm_en",
        sources=("mgsm_en",),
        task="numeric",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="Global-MGSM English (250 grade-school numeric items)",
    ),
    "gsm8k_hi": BenchmarkSpec(
        name="gsm8k_hi",
        sources=("gsm8k_hi",),
        task="numeric",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="nvidia/GSM8K-Hi Hindi grade-school math (1319 numeric items)",
    ),
    "gsm8k_en": BenchmarkSpec(
        name="gsm8k_en",
        sources=("gsm8k_en",),
        task="numeric",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="openai/gsm8k English grade-school math (1319 numeric items)",
    ),
    "m_arena_hard_vi": BenchmarkSpec(
        name="m_arena_hard_vi",
        sources=("m_arena_hard_vi",),
        task="instruction",
        default_splits=("test",),
        submission_csv=False,
        score_gold=False,
        description="m-ArenaHard Vietnamese (500 open-ended prompts; pairwise LLM judge)",
    ),
    "m_arena_hard_hi": BenchmarkSpec(
        name="m_arena_hard_hi",
        sources=("m_arena_hard_hi",),
        task="instruction",
        default_splits=("test",),
        submission_csv=False,
        score_gold=False,
        description="m-ArenaHard Hindi (500 open-ended prompts; pairwise LLM judge)",
    ),
    "mmlu_prox_vi": BenchmarkSpec(
        name="mmlu_prox_vi",
        sources=("mmlu_prox_vi",),
        task="mcq_aj",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="MMLU-ProX Vietnamese test (A–J MCQ)",
    ),
    "mmlu_prox_vi_val": BenchmarkSpec(
        name="mmlu_prox_vi_val",
        sources=("mmlu_prox_vi_val",),
        task="mcq_aj",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="MMLU-ProX Vietnamese validation few-shot pool (70 items)",
    ),
    "mmlu_prox_hi": BenchmarkSpec(
        name="mmlu_prox_hi",
        sources=("mmlu_prox_hi",),
        task="mcq_aj",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="MMLU-ProX Hindi test (A–J MCQ)",
    ),
    "mmlu_prox_hi_val": BenchmarkSpec(
        name="mmlu_prox_hi_val",
        sources=("mmlu_prox_hi_val",),
        task="mcq_aj",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="MMLU-ProX Hindi validation few-shot pool (70 items)",
    ),
    "mmlu_prox_en": BenchmarkSpec(
        name="mmlu_prox_en",
        sources=("mmlu_prox_en",),
        task="mcq_aj",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="MMLU-ProX English test (A–J MCQ)",
    ),
    "mmlu_prox_en_val": BenchmarkSpec(
        name="mmlu_prox_en_val",
        sources=("mmlu_prox_en_val",),
        task="mcq_aj",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="MMLU-ProX English validation few-shot pool (70 items)",
    ),
    "polymath_vi": BenchmarkSpec(
        name="polymath_vi",
        sources=("polymath_vi",),
        task="polymath",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="PolyMath Vietnamese (low/medium/high/top)",
    ),
    "polymath_hi": BenchmarkSpec(
        name="polymath_hi",
        sources=("polymath_hi",),
        task="polymath",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="PolyMath Hindi via PluraMath (low/medium/high/top)",
    ),
    "polymath_en": BenchmarkSpec(
        name="polymath_en",
        sources=("polymath_en",),
        task="polymath",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="PolyMath English (low/medium/high/top)",
    ),
    "vmlu": BenchmarkSpec(
        name="vmlu",
        sources=("vmlu",),
        task="mcq_ae",
        default_splits=("test",),
        submission_csv=True,
        score_gold=True,
        description="VMLU public test: write leaderboard CSV (id,answer); gold is hidden on test",
    ),
    "milu_1_5": BenchmarkSpec(
        name="milu_1_5",
        sources=("milu_1_5",),
        task="mcq_ae",
        default_splits=("test",),
        submission_csv=False,
        score_gold=True,
        description="MILU-1.5 Hindi test (A–D MCQ; gold is in the dataset)",
    ),
    "ifeval": BenchmarkSpec(
        name="ifeval",
        sources=("ifeval",),
        task="instruction",
        default_splits=None,
        submission_csv=False,
        score_gold=False,
        description="English IFEval prompts (generation + L2; prompt-level strict acc stays on eval/model_eval)",
    ),
    "tiny": BenchmarkSpec(
        name="tiny",
        sources=("tiny",),
        task="numeric",
        default_splits=None,
        submission_csv=False,
        score_gold=True,
        description="Bundled synthetic items for plumbing. Not a quality claim.",
    ),
}

def default_prompt_style(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key.startswith("mmlu_prox"):
        return "mmlu-prox"
    if key.startswith("polymath"):
        return "polymath"
    return "contract"


_ALIASES = {
    "tiny": "tiny",
    "mgsm": "mgsm_vi",
    "mgsm-vi": "mgsm_vi",
    "mgsm_vi": "mgsm_vi",
    "global-mgsm-vi": "mgsm_vi",
    "global_mgsm_vi": "mgsm_vi",
    "mgsm-en": "mgsm_en",
    "mgsm_en": "mgsm_en",
    "mgsm-en-en": "mgsm_en",
    "global-mgsm-en": "mgsm_en",
    "global_mgsm_en": "mgsm_en",
    "gsm8k": "gsm8k_en",
    "gsm8k-en": "gsm8k_en",
    "gsm8k_en": "gsm8k_en",
    "gsm8ken": "gsm8k_en",
    "openai-gsm8k": "gsm8k_en",
    "openai_gsm8k": "gsm8k_en",
    "gsm8k-hi": "gsm8k_hi",
    "gsm8k_hi": "gsm8k_hi",
    "gsm8khi": "gsm8k_hi",
    "nvidia-gsm8k-hi": "gsm8k_hi",
    "nvidia_gsm8k_hi": "gsm8k_hi",
    "m-arenahard-vi": "m_arena_hard_vi",
    "m_arenahard_vi": "m_arena_hard_vi",
    "m-arena-hard-vi": "m_arena_hard_vi",
    "m_arena_hard_vi": "m_arena_hard_vi",
    "marenahardvi": "m_arena_hard_vi",
    "m-arenahard-hi": "m_arena_hard_hi",
    "m_arenahard_hi": "m_arena_hard_hi",
    "m-arena-hard-hi": "m_arena_hard_hi",
    "m_arena_hard_hi": "m_arena_hard_hi",
    "marenahardhi": "m_arena_hard_hi",
    "mmlu-prox-hi": "mmlu_prox_hi",
    "mmlu_prox_hi": "mmlu_prox_hi",
    "mmlu-prox-x-hi": "mmlu_prox_hi",
    "mmlu_prox_x_hi": "mmlu_prox_hi",
    "mmluproxhi": "mmlu_prox_hi",
    "mmlu-prox-hi-val": "mmlu_prox_hi_val",
    "mmlu_prox_hi_val": "mmlu_prox_hi_val",
    "mmlu-prox-vi": "mmlu_prox_vi",
    "mmlu_prox_vi": "mmlu_prox_vi",
    "mmlu-prox": "mmlu_prox_vi",
    "mmlu_prox": "mmlu_prox_vi",
    "mmlu-prox-x-vi": "mmlu_prox_vi",
    "mmlu_prox_x_vi": "mmlu_prox_vi",
    "mmluproxvi": "mmlu_prox_vi",
    "mmlu-prox-x": "mmlu_prox_vi",
    "mmlu-prox-en": "mmlu_prox_en",
    "mmlu_prox_en": "mmlu_prox_en",
    "mmlu-prox-x-en": "mmlu_prox_en",
    "mmlu_prox_x_en": "mmlu_prox_en",
    "mmluproxen": "mmlu_prox_en",
    "mmlu-prox-en-val": "mmlu_prox_en_val",
    "mmlu_prox_en_val": "mmlu_prox_en_val",
    "polymath": "polymath_vi",
    "polymath-vi": "polymath_vi",
    "polymath_vi": "polymath_vi",
    "polymath-hi": "polymath_hi",
    "polymath_hi": "polymath_hi",
    "polymath-en": "polymath_en",
    "polymath_en": "polymath_en",
    "pluramath-hi": "polymath_hi",
    "pluramath_hi": "polymath_hi",
    "vmlu": "vmlu",
    "vmlu-test": "vmlu",
    "milu": "milu_1_5",
    "milu-1.5": "milu_1_5",
    "milu_1_5": "milu_1_5",
    "milu-1-5": "milu_1_5",
    "milu_1.5": "milu_1_5",
    "milu-hi": "milu_1_5",
    "milu_hi": "milu_1_5",
    "milu-1.5-hi": "milu_1_5",
    "milu_1_5_hi": "milu_1_5",
    "ifeval": "ifeval",
    "ifeval-en": "ifeval",
    "ifeval_en": "ifeval",
    "google-ifeval": "ifeval",
    "google_ifeval": "ifeval",
    "instruction-following-eval": "ifeval",
    "all": "all",
}


def canonicalize_benchmark(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "_")
    key = key.replace("mmlu-prox-x", "mmlu-prox").replace("mmlu_prox_x", "mmlu_prox")
    if key in _ALIASES:
        return _ALIASES[key]
    collapsed = key.replace("-", "_")
    if collapsed in _ALIASES:
        return _ALIASES[collapsed]
    raise ValueError(
        f"unknown benchmark {name!r}. Choose from: {', '.join(BENCHMARKS)} or 'all'"
    )


def parse_benchmark_list(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("benchmark must not be empty")
    names: list[str] = []
    seen: set[str] = set()
    for part in parts:
        canon = canonicalize_benchmark(part)
        chosen = DEFAULT_RUN_BENCHMARKS if canon == "all" else (canon,)
        for name in chosen:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def spec_for(name: str) -> BenchmarkSpec:
    canon = canonicalize_benchmark(name)
    if canon == "all":
        raise ValueError("resolve 'all' with parse_benchmark_list first")
    return BENCHMARKS[canon]

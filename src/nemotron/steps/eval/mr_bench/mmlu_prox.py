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

"""Official lm-eval MMLU-ProX prompt, 5-shot CoT, and letter extractor."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from nemotron.steps.eval.mr_l2.constants import MCQ_AJ
from nemotron.steps.eval.mr_l2.types import EvalRecord

MMLU_PROX_SHOTS = 5
MMLU_PROX_MAX_TOKENS = 2048
MMLU_PROX_TEMPERATURE = 0.0
MMLU_PROX_TOP_P = 1.0


@dataclass(frozen=True)
class MmluProxLang:
    question: str
    options: str
    answer_open: str
    fewshot_open: str
    description: str
    letter_re: re.Pattern[str]
    stop: tuple[str, ...]
    subjects: dict[str, str]
    letter_extractor: str


# lm_eval/tasks/mmlu_prox/lang_libs.py  lang="vi"
_VI = MmluProxLang(
    question="Câu hỏi:",
    options="Lựa chọn:",
    answer_open="Trả lời: Hãy suy nghĩ từng bước một.",
    fewshot_open="A: Hãy suy nghĩ từng bước một.",
    description=(
        "Dưới đây là câu hỏi trắc nghiệm về {subject} (kèm đáp án). "
        'Vui lòng suy nghĩ từng bước, sau đó kết thúc câu trả lời của bạn bằng "Câu trả lời là (X)", '
        "trong đó X là chữ cái của lựa chọn đúng.\n"
    ),
    letter_re=re.compile(r"Câu trả lời là \(?([ABCDEFGHIJ])\)?"),
    # vi/_vi_template_yaml until, minus the yaml `" "` entry (it would cut CoT at the first space).
    stop=("Q:", "Câu hỏi:", "<|im_end|>"),
    subjects={
        "biology": "Sinh học",
        "business": "Kinh doanh",
        "chemistry": "Hóa học",
        "computer_science": "Khoa học máy tính",
        "economics": "Kinh tế học",
        "engineering": "Kỹ thuật",
        "health": "Sức khỏe",
        "history": "Lịch sử",
        "law": "Luật pháp",
        "math": "Toán học",
        "other": "Khác",
        "philosophy": "Triết học",
        "physics": "Vật lý học",
        "psychology": "Tâm lý học",
    },
    letter_extractor="cau_tra_loi_la",
)

# lm_eval/tasks/mmlu_prox/lang_libs.py  lang="hi" + hi/_hi_template_yaml
_HI = MmluProxLang(
    question="प्रश्न:",
    options="विकल्प:",
    answer_open="उत्तर: चलिए चरण-दर-चरण सोचते हैं।",
    fewshot_open="A: चलिए चरण-दर-चरण सोचते हैं।",
    description=(
        "निम्नलिखित {subject} के बारे में बहुविकल्पीय प्रश्न (उत्तरों के साथ) हैं। "
        'चरण-दर-चरण सोचें और फिर अपने उत्तर को "उत्तर है (X)" के साथ समाप्त करें '
        "जहां X सही विकल्प का अक्षर है।\n"
    ),
    letter_re=re.compile(r"उत्तर है \(?([ABCDEFGHIJ])\)?"),
    stop=("Q:", "प्रश्न:", "<|im_end|>"),
    subjects={
        "biology": "जीव विज्ञान",
        "business": "व्यापार",
        "chemistry": "रसायन विज्ञान",
        "computer_science": "कंप्यूटर विज्ञान",
        "economics": "अर्थशास्त्र",
        "engineering": "इंजीनियरिंग",
        "health": "स्वास्थ्य",
        "history": "इतिहास",
        "law": "कानून",
        "math": "गणित",
        "other": "अन्य",
        "philosophy": "दर्शनशास्त्र",
        "physics": "भौतिकी",
        "psychology": "मनोविज्ञान",
    },
    letter_extractor="uttar_hai",
)

# lm_eval/tasks/mmlu_prox/lang_libs.py  lang="en" + en/_en_template_yaml
_EN = MmluProxLang(
    question="Question:",
    options="Options:",
    answer_open="Answer: Let's think step by step.",
    fewshot_open="A: Let's think step by step.",
    description=(
        "The following are multiple choice questions (with answers) about {subject}. "
        'Think step by step and then finish your answer with "the answer is (X)" '
        "where X is the correct letter choice.\n"
    ),
    letter_re=re.compile(r"the answer is \(?([ABCDEFGHIJ])\)?"),
    stop=("Q:", "Question:", "<|im_end|>"),
    subjects={
        "biology": "biology",
        "business": "business",
        "chemistry": "chemistry",
        "computer_science": "computer science",
        "economics": "economics",
        "engineering": "engineering",
        "health": "health",
        "history": "history",
        "law": "law",
        "math": "math",
        "other": "other",
        "philosophy": "philosophy",
        "physics": "physics",
        "psychology": "psychology",
    },
    letter_extractor="the_answer_is",
)

LANGS = {"vi": _VI, "hi": _HI, "en": _EN}

MMLU_PROX_SOURCES = {
    "mmlu_prox_vi": "vi",
    "mmlu_prox_vi_val": "vi",
    "mmlu_prox_hi": "hi",
    "mmlu_prox_hi_val": "hi",
    "mmlu_prox_en": "en",
    "mmlu_prox_en_val": "en",
}

# Back-compat aliases for Vietnamese tests/imports.
MMLU_PROX_STOP = _VI.stop
_QUESTION = _VI.question
_OPTIONS = _VI.options
_ANSWER_OPEN = _VI.answer_open
_FEWSHOT_OPEN = _VI.fewshot_open
_SUBJECT_VI = _VI.subjects
_DESCRIPTION = _VI.description
_OFFICIAL_LETTER = _VI.letter_re


def mmlu_prox_lang(name: str | None) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key in MMLU_PROX_SOURCES:
        return MMLU_PROX_SOURCES[key]
    if "mmlu_prox_hi" in key or key.endswith("_hi"):
        return "hi"
    if "mmlu_prox_en" in key or key.endswith("_en"):
        return "en"
    return "vi"


def mmlu_prox_stop(lang: str = "vi") -> tuple[str, ...]:
    return LANGS[lang].stop


def mmlu_prox_letter_extractor(lang: str = "vi") -> str:
    return LANGS[lang].letter_extractor


def is_mmlu_prox_benchmark(name: str | None) -> bool:
    return (name or "") in MMLU_PROX_SOURCES


def is_mmlu_prox_official(prompt_style: str | None, source: str | None = None) -> bool:
    if prompt_style != "mmlu-prox":
        return False
    if source is None:
        return True
    return source in MMLU_PROX_SOURCES


def subject_name(stratum: str, lang: str = "vi") -> str:
    key = (stratum or "").strip().lower().replace(" ", "_")
    return LANGS[lang].subjects.get(key, stratum or "unknown")


def subject_vi(stratum: str) -> str:
    return subject_name(stratum, "vi")


def format_mmlu_prox_example(
    question: str,
    options: Sequence[str | None] | None,
    *,
    including_answer: bool,
    cot_content: str | None = None,
    lang: str = "vi",
) -> str:
    """Mirror lm_eval mmlu_prox/<lang>/utils.py format_cot_example."""
    lib = LANGS[lang]
    prompt = f"{lib.question}\n{(question or '')}\n{lib.options}\n"
    for i, opt in enumerate(options or ()):
        if i >= 10:
            break
        if opt is not None:
            prompt += f"{MCQ_AJ[i]}. {opt}\n"
    if including_answer:
        cot = (cot_content or "").replace(lib.fewshot_open, lib.answer_open)
        return prompt + cot + "\n\n"
    return prompt + lib.answer_open


def format_mmlu_prox_prompt(
    record: EvalRecord,
    fewshots: Sequence[EvalRecord],
    lang: str | None = None,
) -> str:
    lang = lang or mmlu_prox_lang(record.source)
    desc = LANGS[lang].description.format(subject=subject_name(record.stratum, lang))
    parts = [desc]
    for ex in fewshots:
        parts.append(
            format_mmlu_prox_example(
                ex.question,
                ex.options,
                including_answer=True,
                cot_content=(ex.extras or {}).get("cot_content"),
                lang=lang,
            )
        )
    parts.append(
        format_mmlu_prox_example(
            record.question,
            record.options,
            including_answer=False,
            lang=lang,
        )
    )
    return "".join(parts)


def parse_mmlu_prox_official(text: str, lang: str = "vi") -> str:
    match = LANGS[lang].letter_re.search(text or "")
    return match.group(1).upper() if match else ""


@lru_cache(maxsize=8)
def load_mmlu_prox_fewshots(
    data_root: str, k: int = MMLU_PROX_SHOTS, lang: str = "vi"
) -> dict[str, tuple[EvalRecord, ...]]:
    """Validation rows in parquet order; first_n per category, like lm-eval."""
    from nemotron.steps.eval.mr_bench.sources import iter_mmlu_records

    pool: dict[str, list[EvalRecord]] = defaultdict(list)
    for rec in iter_mmlu_records(Path(data_root), validation=True, lang=lang):
        bucket = pool[rec.stratum]
        if len(bucket) < k:
            bucket.append(rec)
    return {key: tuple(rows) for key, rows in pool.items()}

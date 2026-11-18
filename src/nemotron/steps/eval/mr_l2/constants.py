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

from nemotron.steps.eval.mr_l2.types import SourceSpec, TaskKind

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

TOKEN_BUDGET = 32_768
IN_LANGUAGE_THRESHOLD = 0.95
SPLIT_SEED = 20261102
FROZEN_DEV_N = 1_000
COPY_MIN_CHARS = 12
LOOP_NGRAM = 4
LOOP_MIN_COUNT = 8
LOOP_COVERAGE = 0.25
LOOP_CONSECUTIVE = 6

SAMPLING: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": TOKEN_BUDGET,
}

FORCED_OPENERS = {
    "vi": "Tôi sẽ giải quyết từng bước.",
    "en": "I will solve this step by step.",
    "hi": "यह समस्या को हल करने की विचार प्रक्रिया है:",
}

# Fallback when parquet `instruction` / `answer_prefix` are missing.
# Must match CohereLabs/global-mgsm config `vi` (same text on every test row).
MGSM_VI_INSTRUCTION = (
    "Giải bài toán này. Giải thích các bước giải trước khi viết đáp án cuối cùng "
    'vào dòng cuối cùng dưới dạng "Đáp án:". Không thêm bất kỳ đáp án nào khác '
    'ngoài đáp án đúng sau "Đáp án:".'
)
MGSM_VI_ANSWER_PREFIX = "Đáp án"

# CohereLabs/global-mgsm config `en` (same text on every English test row).
MGSM_EN_INSTRUCTION = (
    "Solve this math problem. Give the reasoning steps before giving the final "
    'answer on the last line by itself in the format of "Answer:". Do not add '
    'anything other than the integer answer after "Answer:".'
)
MGSM_EN_ANSWER_PREFIX = "Answer"

# Official GSM8K-Hi lm-eval doc_to_text (nvidia/GSM8K-Hi EVAL.md).
GSM8K_HI_INSTRUCTION = (
    "नीचे एक गणित का प्रश्न है। मैं चाहता हूँ कि आप पहले उत्तर तक पहुँचने के लिए आवश्यक "
    "चरणों के बारे में सोचें, फिर अपने उत्तर को \"#### \" के साथ समाप्त करें और उसके बाद "
    "उत्तर लिखें। उदाहरण के लिए, यदि उत्तर 42 है, तो आपके उत्तर को \"#### 42\" "
    "(उद्धरण चिह्नों के बिना) के साथ समाप्त होना चाहिए।"
)
GSM8K_HI_ANSWER_PREFIX = "####"

# English GSM8K (openai/gsm8k) — same #### closer as GSM8K-Hi EVAL.md.
GSM8K_EN_INSTRUCTION = (
    "Below is a math question. I want you to first think about the steps needed "
    "to reach the answer, then end your answer with \"#### \" followed by the "
    "answer. For example, if the answer is 42, your answer should end with "
    "\"#### 42\" (without quotes)."
)
GSM8K_EN_ANSWER_PREFIX = "####"

# 0-shot MMLU-ProX-vi answer format. Official lm-eval uses 5-shot and
# "Câu trả lời là (X)"; we keep 0-shot but require a terminal choice line so
# the letter parser is not poisoned by later option recaps.
MMLU_PROX_VI_INSTRUCTION = (
    "Dưới đây là câu hỏi trắc nghiệm. Hãy suy nghĩ từng bước, "
    'sau đó kết thúc câu trả lời bằng "Đáp án đúng là X" '
    "trong đó X là chữ cái (A–J) của lựa chọn đúng. "
    "Không thêm bất kỳ nội dung nào khác sau chữ cái đó."
)
MMLU_PROX_VI_ANSWER_PREFIX = "Đáp án đúng là"

MMLU_PROX_HI_INSTRUCTION = (
    "नीचे एक बहुविकल्पीय प्रश्न दिया गया है। चरण-दर-चरण सोचें, "
    'फिर अपने उत्तर को "उत्तर है X" के साथ समाप्त करें '
    "जहाँ X सही विकल्प का अक्षर (A–J) है। "
    "उस अक्षर के बाद कोई अन्य सामग्री न जोड़ें।"
)
MMLU_PROX_HI_ANSWER_PREFIX = "उत्तर है"

MMLU_PROX_EN_INSTRUCTION = (
    "The following is a multiple choice question. Think step by step, "
    'then finish your answer with "The answer is X" '
    "where X is the letter (A–J) of the correct choice. "
    "Do not add any other content after that letter."
)
MMLU_PROX_EN_ANSWER_PREFIX = "The answer is"

# 0-shot MILU-1.5 Hindi (A–D). Official lm-eval is 5-shot greedy; native
# reasoning uses the same Hindi closer as MMLU-ProX-hi so the letter parser
# is not poisoned by later option recaps.
MILU_1_5_INSTRUCTION = (
    "नीचे एक बहुविकल्पीय प्रश्न दिया गया है। चरण-दर-चरण सोचें, "
    'फिर अपने उत्तर को "उत्तर है X" के साथ समाप्त करें '
    "जहाँ X सही विकल्प का अक्षर (A–D) है। "
    "उस अक्षर के बाद कोई अन्य सामग्री न जोड़ें।"
)
MILU_1_5_ANSWER_PREFIX = "उत्तर है"

MCQ_AJ = tuple("ABCDEFGHIJ")
MCQ_AE = tuple("ABCDE")

GOLD_FIELDS = frozenset(
    {
        "gold",
        "answer",
        "answer_number",
        "answer_index",
        "cot_content",
        "equation_solution",
        "canonical_gold",
    }
)

EVAL_SOURCE_NAMES = (
    "mgsm_vi",
    "mgsm_en",
    "gsm8k_hi",
    "gsm8k_en",
    "m_arena_hard_hi",
    "m_arena_hard_vi",
    "mmlu_prox_vi",
    "mmlu_prox_vi_val",
    "mmlu_prox_hi",
    "mmlu_prox_hi_val",
    "mmlu_prox_en",
    "mmlu_prox_en_val",
    "polymath_vi",
    "polymath_hi",
    "polymath_en",
    "vmlu",
    "milu_1_5",
    "ifeval",
)

SOURCES: dict[str, SourceSpec] = {
    "mgsm_vi": SourceSpec("mgsm_vi", "numeric", "eval", "vi"),
    "mgsm_en": SourceSpec("mgsm_en", "numeric", "eval", "en"),
    "gsm8k_hi": SourceSpec("gsm8k_hi", "numeric", "eval", "hi"),
    "gsm8k_en": SourceSpec("gsm8k_en", "numeric", "eval", "en"),
    "m_arena_hard_hi": SourceSpec("m_arena_hard_hi", "instruction", "eval", "hi"),
    "m_arena_hard_vi": SourceSpec("m_arena_hard_vi", "instruction", "eval", "vi"),
    "mmlu_prox_vi": SourceSpec("mmlu_prox_vi", "mcq_aj", "eval", "vi"),
    "mmlu_prox_vi_val": SourceSpec("mmlu_prox_vi_val", "mcq_aj", "eval", "vi"),
    "mmlu_prox_hi": SourceSpec("mmlu_prox_hi", "mcq_aj", "eval", "hi"),
    "mmlu_prox_hi_val": SourceSpec("mmlu_prox_hi_val", "mcq_aj", "eval", "hi"),
    "mmlu_prox_en": SourceSpec("mmlu_prox_en", "mcq_aj", "eval", "en"),
    "mmlu_prox_en_val": SourceSpec("mmlu_prox_en_val", "mcq_aj", "eval", "en"),
    "polymath_vi": SourceSpec("polymath_vi", "polymath", "eval", "vi"),
    "polymath_hi": SourceSpec("polymath_hi", "polymath", "eval", "hi"),
    "polymath_en": SourceSpec("polymath_en", "polymath", "eval", "en"),
    "vmlu": SourceSpec("vmlu", "mcq_ae", "eval", "vi"),
    "milu_1_5": SourceSpec("milu_1_5", "mcq_ae", "eval", "hi"),
    "ifeval": SourceSpec("ifeval", "instruction", "eval", "en"),
    "gsm8k_train_vi": SourceSpec("gsm8k_train_vi", "numeric", "train", "vi"),
    "tinyaya_mr": SourceSpec("tinyaya_mr", "instruction", "train", "und"),
}

TASK_BY_SOURCE: dict[str, TaskKind] = {name: spec.task for name, spec in SOURCES.items()}


def bench_language(name: str | None) -> str:
    raw = (name or "").strip().lower().replace("-", "_")
    spec = SOURCES.get(raw)
    if spec is not None:
        return spec.language
    key = f"_{raw}_"
    if "_hi_" in key:
        return "hi"
    if "_en_" in key:
        return "en"
    return "vi"

SAMPLE_LOG_FIELDS = (
    "item_id",
    "source",
    "split",
    "mode",
    "path",
    "task",
    "prompt_language",
    "requested_answer_language",
    "thinking_language_target",
    "user_message",
    "thinking_prefix",
    "raw_output",
    "thinking",
    "answer_span",
    "parsed_answer",
    "parser_answer",
    "parser_correct",
    "thinking_lang_share",
    "in_language_95",
    "answer_language",
    "closed",
    "truncated",
    "loop",
    "n_tokens",
    "sampling_hash",
    "correct",
    "gold",
    "hard_gate",
)

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

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from nemotron.steps.eval.mr_l2.constants import (
    EVAL_SOURCE_NAMES,
    GSM8K_EN_ANSWER_PREFIX,
    GSM8K_EN_INSTRUCTION,
    GSM8K_HI_ANSWER_PREFIX,
    GSM8K_HI_INSTRUCTION,
    MCQ_AE,
    MCQ_AJ,
    MGSM_EN_ANSWER_PREFIX,
    MGSM_EN_INSTRUCTION,
    MGSM_VI_ANSWER_PREFIX,
    MGSM_VI_INSTRUCTION,
    MILU_1_5_ANSWER_PREFIX,
    MILU_1_5_INSTRUCTION,
    MMLU_PROX_EN_ANSWER_PREFIX,
    MMLU_PROX_EN_INSTRUCTION,
    MMLU_PROX_HI_ANSWER_PREFIX,
    MMLU_PROX_HI_INSTRUCTION,
    MMLU_PROX_VI_ANSWER_PREFIX,
    MMLU_PROX_VI_INSTRUCTION,
    SOURCES,
)
from nemotron.steps.eval.mr_bench.ids import (
    gsm8k_hi_id,
    gsm8k_id,
    ifeval_id,
    m_arena_hard_id,
    mgsm_id,
    milu_id,
    mmlu_id,
    parse_item_id,
    polymath_id,
    vmlu_id,
)
from nemotron.steps.eval.mr_l2.parser import extract_gsm8k_gold
from nemotron.steps.eval.mr_l2.types import EvalRecord, ItemRef

STEP_DIR = Path(__file__).resolve().parent

_VMLU_FILES = (
    ("train.jsonl", "train"),
    ("valid.jsonl", "validation"),
    ("test.jsonl", "test"),
)
_VMLU_CHOICE_PREFIX = re.compile(r"^[A-Ea-e][.)]\s*")

_MMLU_TEST = Path("mmlu_prox/vi/test-00000-of-00001.parquet")
_MMLU_VAL = Path("mmlu_prox/vi/validation-00000-of-00001.parquet")
_M_ARENA_HARD = Path("m_arena_hard")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _present_options(row: dict[str, Any], n: int = 10) -> tuple[str | None, ...]:
    opts: list[str | None] = []
    for i in range(n):
        key = f"option_{i}"
        if key not in row:
            break
        val = row[key]
        opts.append(None if val is None else str(val))
    return tuple(opts)


def _item_ref(record: EvalRecord) -> ItemRef:
    return ItemRef(
        item_id=record.item_id,
        source=record.source,
        stratum=record.stratum,
        split=record.split,
        extras={"question": record.question},
    )


def _mmlu_gold_letter(row: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Trust publisher letter, not answer_index. Record mismatches in extras."""
    letter = str(row.get("answer") or "").strip().upper()[:1]
    extras: dict[str, str] = {}
    idx = row.get("answer_index")
    if idx is not None and letter:
        try:
            index_letter = MCQ_AJ[int(idx)]
        except (TypeError, ValueError, IndexError):
            index_letter = ""
        if index_letter and index_letter != letter:
            extras["answer_index_mismatch"] = (
                f"letter={letter},index={idx},index_letter={index_letter}"
            )
    return letter, extras


def iter_mgsm_records(data_root: Path, *, lang: str = "vi") -> Iterator[EvalRecord]:
    path = data_root / "global_mgsm" / lang / "test.parquet"
    if not path.is_file():
        return
    source = f"mgsm_{lang}"
    spec = SOURCES[source]
    if lang == "en":
        fallback_instruction, fallback_prefix = MGSM_EN_INSTRUCTION, MGSM_EN_ANSWER_PREFIX
    else:
        fallback_instruction, fallback_prefix = MGSM_VI_INSTRUCTION, MGSM_VI_ANSWER_PREFIX
    for row in _read_parquet(path):
        question = str(row["question"])
        yield EvalRecord(
            item_id=mgsm_id(question, lang=lang),
            source=source,
            task=spec.task,
            question=question,
            gold=str(row["answer"]).strip(),
            prompt_language=lang,
            stratum="grade-school",
            instruction=str(row.get("instruction") or "").strip() or fallback_instruction,
            answer_prefix=str(row.get("answer_prefix") or "").strip() or fallback_prefix,
        )


def iter_gsm8k_hi_records(data_root: Path) -> Iterator[EvalRecord]:
    path = data_root / "gsm8k_hi" / "test.jsonl"
    if not path.is_file():
        return
    spec = SOURCES["gsm8k_hi"]
    for row in _read_jsonl(path):
        question = str(row["question"])
        yield EvalRecord(
            item_id=gsm8k_hi_id(question),
            source=spec.name,
            task=spec.task,
            question=question,
            gold=extract_gsm8k_gold(str(row.get("answer") or "")),
            prompt_language="hi",
            stratum="grade-school",
            instruction=str(row.get("instruction") or "").strip() or GSM8K_HI_INSTRUCTION,
            answer_prefix=str(row.get("answer_prefix") or "").strip() or GSM8K_HI_ANSWER_PREFIX,
        )


def _gsm8k_en_rows(data_root: Path) -> Iterator[dict[str, Any]]:
    jsonl = data_root / "gsm8k_en" / "test.jsonl"
    if jsonl.is_file():
        yield from _read_jsonl(jsonl)
        return
    for parquet in (
        data_root / "gsm8k_en" / "test.parquet",
        data_root / "gsm8k_en" / "main" / "test-00000-of-00001.parquet",
    ):
        if parquet.is_file():
            yield from _read_parquet(parquet)
            return


def iter_gsm8k_en_records(data_root: Path) -> Iterator[EvalRecord]:
    spec = SOURCES["gsm8k_en"]
    for row in _gsm8k_en_rows(data_root):
        question = str(row["question"])
        yield EvalRecord(
            item_id=gsm8k_id(question, lang="en"),
            source=spec.name,
            task=spec.task,
            question=question,
            gold=extract_gsm8k_gold(str(row.get("answer") or "")),
            prompt_language="en",
            stratum="grade-school",
            instruction=str(row.get("instruction") or "").strip() or GSM8K_EN_INSTRUCTION,
            answer_prefix=str(row.get("answer_prefix") or "").strip() or GSM8K_EN_ANSWER_PREFIX,
        )


def iter_m_arena_hard_records(
    data_root: Path, *, lang: str = "vi"
) -> Iterator[EvalRecord]:
    path = data_root / _M_ARENA_HARD / lang / "test-00000-of-00001.parquet"
    if not path.is_file():
        return
    spec = SOURCES[f"m_arena_hard_{lang}"]
    for row in _read_parquet(path):
        question_id = str(row["question_id"]).strip()
        yield EvalRecord(
            item_id=m_arena_hard_id(question_id, lang=lang),
            source=spec.name,
            task=spec.task,
            question=str(row["prompt"]),
            gold="",
            prompt_language=lang,
            stratum=str(row.get("cluster") or "unknown"),
            split="test",
            extras={
                "question_id": question_id,
                "category": str(row.get("category") or ""),
            },
        )


def iter_mmlu_records(data_root: Path, *, validation: bool, lang: str = "vi") -> Iterator[EvalRecord]:
    filename = "validation-00000-of-00001.parquet" if validation else "test-00000-of-00001.parquet"
    path = data_root / "mmlu_prox" / lang / filename
    if not path.is_file():
        return
    source = f"mmlu_prox_{lang}_val" if validation else f"mmlu_prox_{lang}"
    split_name = "fewshot" if validation else None
    spec = SOURCES[source]
    if lang == "hi":
        instruction, prefix = MMLU_PROX_HI_INSTRUCTION, MMLU_PROX_HI_ANSWER_PREFIX
    elif lang == "en":
        instruction, prefix = MMLU_PROX_EN_INSTRUCTION, MMLU_PROX_EN_ANSWER_PREFIX
    else:
        instruction, prefix = MMLU_PROX_VI_INSTRUCTION, MMLU_PROX_VI_ANSWER_PREFIX
    for row in _read_parquet(path):
        qid = int(row["question_id"])
        gold, extras = _mmlu_gold_letter(row)
        if validation:
            extras["cot_content"] = str(row.get("cot_content") or "")
        yield EvalRecord(
            item_id=mmlu_id(qid, validation=validation, lang=lang),
            source=source,
            task=spec.task,
            question=str(row["question"]),
            gold=gold,
            prompt_language=lang,
            stratum=str(row.get("category") or "unknown"),
            options=_present_options(row, 10),
            split=split_name,
            extras=extras,
            instruction=instruction,
            answer_prefix=prefix,
        )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if raw:
                yield json.loads(raw)


def _vmlu_options(choices: Any) -> tuple[str | None, ...]:
    if not choices:
        return ()
    opts: list[str | None] = []
    for item in choices:
        text = _VMLU_CHOICE_PREFIX.sub("", str(item).strip())
        opts.append(text or None)
    return tuple(opts)


_MILU_TARGET = {
    "option1": "A",
    "option2": "B",
    "option3": "C",
    "option4": "D",
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
}


def _milu_gold(target: Any) -> str:
    raw = str(target or "").strip()
    key = raw.lower().replace(" ", "")
    if key in _MILU_TARGET:
        return _MILU_TARGET[key]
    letter = raw.upper()[:1]
    if letter in MCQ_AE[:4]:
        return letter
    return ""


def _milu_parquet(data_root: Path, split: str) -> Path | None:
    candidates = (
        data_root / "milu_1_5" / "Hindi" / f"{split}-00000-of-00001.parquet",
        data_root / "milu_1_5" / "hi" / f"{split}-00000-of-00001.parquet",
        data_root / "milu_1_5" / f"{split}-00000-of-00001.parquet",
    )
    for path in candidates:
        if path.is_file():
            return path
    for folder in (
        data_root / "milu_1_5" / "Hindi",
        data_root / "milu_1_5" / "hi",
        data_root / "milu_1_5",
    ):
        if folder.is_dir():
            matches = sorted(folder.glob(f"{split}-*.parquet"))
            if matches:
                return matches[0]
    return None


def iter_milu_records(data_root: Path) -> Iterator[EvalRecord]:
    spec = SOURCES["milu_1_5"]
    for split in ("test", "validation"):
        path = _milu_parquet(data_root, split)
        if path is None:
            continue
        for row in _read_parquet(path):
            options = tuple(
                str(row.get(key) or "").strip() for key in ("option1", "option2", "option3", "option4")
            )
            question = str(row.get("question") or "")
            yield EvalRecord(
                item_id=milu_id(question, options),
                source="milu_1_5",
                task=spec.task,
                question=question,
                gold=_milu_gold(row.get("target")),
                prompt_language="hi",
                stratum=str(row.get("subject") or row.get("domain") or "unknown"),
                options=options,
                split=split,
                extras={
                    "domain": str(row.get("domain") or ""),
                    "subject": str(row.get("subject") or ""),
                    "is_translated": str(row.get("is_translated") or ""),
                },
                instruction=MILU_1_5_INSTRUCTION,
                answer_prefix=MILU_1_5_ANSWER_PREFIX,
            )


def iter_vmlu_records(data_root: Path) -> Iterator[EvalRecord]:
    root = data_root / "vmlu"
    spec = SOURCES["vmlu"]
    for filename, split in _VMLU_FILES:
        path = root / filename
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            native = str(row.get("id") or "").strip()
            if not native:
                continue
            gold = str(row.get("answer") or "").strip().upper()[:1]
            yield EvalRecord(
                item_id=vmlu_id(native),
                source="vmlu",
                task=spec.task,
                question=str(row["question"]),
                gold=gold,
                prompt_language="vi",
                stratum=native.split("-", 1)[0],
                options=_vmlu_options(row.get("choices")),
                split=split,
            )


def iter_polymath_records(data_root: Path, *, lang: str = "vi") -> Iterator[EvalRecord]:
    source = f"polymath_{lang}"
    root = data_root / "polymath" / lang
    if not root.is_dir() or source not in SOURCES:
        return
    spec = SOURCES[source]
    for level in ("low", "medium", "high", "top"):
        path = root / f"{level}.parquet"
        if not path.is_file():
            continue
        for row in _read_parquet(path):
            native = str(row["id"])
            yield EvalRecord(
                item_id=polymath_id(native, lang=lang),
                source=source,
                task=spec.task,
                question=str(row["question"]),
                gold=str(row["answer"]).strip(),
                prompt_language=lang,
                stratum=level,
            )


def iter_mgsm(data_root: Path) -> Iterator[ItemRef]:
    for record in iter_mgsm_records(data_root):
        yield _item_ref(record)


def iter_m_arena_hard(data_root: Path, *, lang: str = "vi") -> Iterator[ItemRef]:
    for record in iter_m_arena_hard_records(data_root, lang=lang):
        yield _item_ref(record)


def iter_mmlu(data_root: Path, *, validation: bool) -> Iterator[ItemRef]:
    for record in iter_mmlu_records(data_root, validation=validation):
        yield _item_ref(record)


def iter_polymath(data_root: Path) -> Iterator[ItemRef]:
    for record in iter_polymath_records(data_root):
        yield _item_ref(record)


def iter_vmlu(data_root: Path) -> Iterator[ItemRef]:
    for record in iter_vmlu_records(data_root):
        yield _item_ref(record)


def iter_milu(data_root: Path) -> Iterator[ItemRef]:
    for record in iter_milu_records(data_root):
        yield _item_ref(record)


def iter_ifeval_records(data_root: Path) -> Iterator[EvalRecord]:
    path = data_root / "ifeval" / "ifeval_input_data.jsonl"
    if not path.is_file():
        return
    spec = SOURCES["ifeval"]
    for row in _read_jsonl(path):
        key = row.get("key")
        prompt = str(row.get("prompt") or "")
        instruction_id_list = row.get("instruction_id_list") or []
        kwargs = row.get("kwargs") or []
        first = str(instruction_id_list[0]) if instruction_id_list else "ifeval"
        stratum = first.split(":", 1)[0] if first else "ifeval"
        yield EvalRecord(
            item_id=ifeval_id(key),
            source=spec.name,
            task=spec.task,
            question=prompt,
            gold="",
            prompt_language="en",
            stratum=stratum,
            split="train",
            extras={
                "key": str(key),
                "instruction_id_list": json.dumps(instruction_id_list, ensure_ascii=False),
                "kwargs": json.dumps(kwargs, ensure_ascii=False),
            },
        )


def iter_tiny_records(path: Path | None = None) -> Iterator[EvalRecord]:
    target = path or (STEP_DIR / "config" / "tiny_items.jsonl")
    if not target.is_file():
        return
    spec = SOURCES["mgsm_vi"]
    for row in _read_jsonl(target):
        yield EvalRecord(
            item_id=str(row.get("item_id") or "tiny:unknown"),
            source=str(row.get("source") or "tiny"),
            task=row.get("task") or spec.task,  # type: ignore[arg-type]
            question=str(row.get("question") or ""),
            gold=str(row.get("gold") or ""),
            prompt_language=str(row.get("prompt_language") or "vi"),
            stratum=str(row.get("stratum") or "tiny"),
            options=tuple(row["options"]) if row.get("options") else None,
            split=str(row.get("split") or "smoke"),
            instruction=row.get("instruction"),
            answer_prefix=row.get("answer_prefix"),
        )


def iter_source_records(data_root: Path, source: str) -> Iterator[EvalRecord]:
    if source == "tiny":
        bundled = STEP_DIR / "config" / "tiny_items.jsonl"
        local = data_root / "tiny_items.jsonl"
        yield from iter_tiny_records(local if local.is_file() else bundled)
        return
    if source == "mgsm_vi":
        yield from iter_mgsm_records(data_root, lang="vi")
    elif source == "mgsm_en":
        yield from iter_mgsm_records(data_root, lang="en")
    elif source == "gsm8k_hi":
        yield from iter_gsm8k_hi_records(data_root)
    elif source == "gsm8k_en":
        yield from iter_gsm8k_en_records(data_root)
    elif source == "m_arena_hard_vi":
        yield from iter_m_arena_hard_records(data_root, lang="vi")
    elif source == "m_arena_hard_hi":
        yield from iter_m_arena_hard_records(data_root, lang="hi")
    elif source == "mmlu_prox_vi":
        yield from iter_mmlu_records(data_root, validation=False, lang="vi")
    elif source == "mmlu_prox_vi_val":
        yield from iter_mmlu_records(data_root, validation=True, lang="vi")
    elif source == "mmlu_prox_hi":
        yield from iter_mmlu_records(data_root, validation=False, lang="hi")
    elif source == "mmlu_prox_hi_val":
        yield from iter_mmlu_records(data_root, validation=True, lang="hi")
    elif source == "mmlu_prox_en":
        yield from iter_mmlu_records(data_root, validation=False, lang="en")
    elif source == "mmlu_prox_en_val":
        yield from iter_mmlu_records(data_root, validation=True, lang="en")
    elif source == "polymath_vi":
        yield from iter_polymath_records(data_root, lang="vi")
    elif source == "polymath_hi":
        yield from iter_polymath_records(data_root, lang="hi")
    elif source == "polymath_en":
        yield from iter_polymath_records(data_root, lang="en")
    elif source == "vmlu":
        yield from iter_vmlu_records(data_root)
    elif source == "milu_1_5":
        yield from iter_milu_records(data_root)
    elif source == "ifeval":
        yield from iter_ifeval_records(data_root)
    else:
        raise ValueError(f"unknown source {source!r}")


def effective_split(record: EvalRecord) -> str:
    if record.split:
        return record.split
    return "test"


def load_eval_records(
    data_root: str | Path,
    *,
    ids: Iterable[str] | None = None,
    sources: Iterable[str] | None = None,
    splits: Iterable[str] | None = None,
) -> list[EvalRecord]:
    """Load eval rows with gold. Caller must not write the result into manifests/."""
    root = Path(data_root)
    if sources is not None:
        wanted_sources = tuple(sources)
    elif ids is not None:
        inferred: list[str] = []
        seen: set[str] = set()
        for item_id in ids:
            try:
                src, _ = parse_item_id(item_id)
            except ValueError:
                inferred = list(EVAL_SOURCE_NAMES)
                break
            if src not in seen and src in SOURCES:
                inferred.append(src)
                seen.add(src)
        wanted_sources = tuple(inferred) if inferred else EVAL_SOURCE_NAMES
    else:
        wanted_sources = EVAL_SOURCE_NAMES
    records: list[EvalRecord] = []
    for source in wanted_sources:
        records.extend(iter_source_records(root, source))
    if splits is not None:
        wanted_splits = {str(s) for s in splits}
        records = [r for r in records if effective_split(r) in wanted_splits]
    if ids is not None:
        wanted = set(ids)
        records = [r for r in records if r.item_id in wanted]
    records.sort(key=lambda r: r.item_id)
    return records


def load_eval_items(data_root: str | Path) -> list[ItemRef]:
    root = Path(data_root)
    items: list[ItemRef] = []
    items.extend(iter_mgsm(root))
    items.extend(iter_m_arena_hard(root, lang="vi"))
    items.extend(iter_m_arena_hard(root, lang="hi"))
    items.extend(iter_mmlu(root, validation=False))
    items.extend(iter_mmlu(root, validation=True))
    items.extend(iter_polymath(root))
    items.extend(iter_vmlu(root))
    items.extend(iter_milu(root))
    items.sort(key=lambda r: r.item_id)
    return items

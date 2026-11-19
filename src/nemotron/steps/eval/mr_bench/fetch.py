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

"""Download Native-reasoning eval benches from Hugging Face into data_root."""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# source name -> (hub repo, ((remote_path, local_path), ...))
HF_DATASETS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "mgsm_vi": ("CohereLabs/global-mgsm", (("vi/test.parquet", "data/global_mgsm/vi/test.parquet"),)),
    "mgsm_en": ("CohereLabs/global-mgsm", (("en/test.parquet", "data/global_mgsm/en/test.parquet"),)),
    "gsm8k_hi": ("nvidia/GSM8K-Hi", (("test.jsonl", "data/gsm8k_hi/test.jsonl"),)),
    "gsm8k_en": (
        "openai/gsm8k",
        (("main/test-00000-of-00001.parquet", "data/gsm8k_en/main/test-00000-of-00001.parquet"),),
    ),
    "mmlu_prox_vi": (
        "li-lab/MMLU-ProX",
        (
            ("vi/test-00000-of-00001.parquet", "data/mmlu_prox/vi/test-00000-of-00001.parquet"),
            ("vi/validation-00000-of-00001.parquet", "data/mmlu_prox/vi/validation-00000-of-00001.parquet"),
        ),
    ),
    "mmlu_prox_vi_val": (
        "li-lab/MMLU-ProX",
        (("vi/validation-00000-of-00001.parquet", "data/mmlu_prox/vi/validation-00000-of-00001.parquet"),),
    ),
    "mmlu_prox_hi": (
        "li-lab/MMLU-ProX",
        (
            ("hi/test-00000-of-00001.parquet", "data/mmlu_prox/hi/test-00000-of-00001.parquet"),
            ("hi/validation-00000-of-00001.parquet", "data/mmlu_prox/hi/validation-00000-of-00001.parquet"),
        ),
    ),
    "mmlu_prox_hi_val": (
        "li-lab/MMLU-ProX",
        (("hi/validation-00000-of-00001.parquet", "data/mmlu_prox/hi/validation-00000-of-00001.parquet"),),
    ),
    "mmlu_prox_en": (
        "li-lab/MMLU-ProX",
        (
            ("en/test-00000-of-00001.parquet", "data/mmlu_prox/en/test-00000-of-00001.parquet"),
            ("en/validation-00000-of-00001.parquet", "data/mmlu_prox/en/validation-00000-of-00001.parquet"),
        ),
    ),
    "mmlu_prox_en_val": (
        "li-lab/MMLU-ProX",
        (("en/validation-00000-of-00001.parquet", "data/mmlu_prox/en/validation-00000-of-00001.parquet"),),
    ),
    "polymath_vi": (
        "Qwen/PolyMath",
        (
            ("vi/low.parquet", "data/polymath/vi/low.parquet"),
            ("vi/medium.parquet", "data/polymath/vi/medium.parquet"),
            ("vi/high.parquet", "data/polymath/vi/high.parquet"),
            ("vi/top.parquet", "data/polymath/vi/top.parquet"),
        ),
    ),
    "polymath_en": (
        "Qwen/PolyMath",
        (
            ("en/low.parquet", "data/polymath/en/low.parquet"),
            ("en/medium.parquet", "data/polymath/en/medium.parquet"),
            ("en/high.parquet", "data/polymath/en/high.parquet"),
            ("en/top.parquet", "data/polymath/en/top.parquet"),
        ),
    ),
    "polymath_hi": (
        "tum-nlp/PluraMath",
        (
            ("hi/low-00000-of-00001.parquet", "data/polymath/hi/low.parquet"),
            ("hi/medium-00000-of-00001.parquet", "data/polymath/hi/medium.parquet"),
            ("hi/high-00000-of-00001.parquet", "data/polymath/hi/high.parquet"),
            ("hi/top-00000-of-00001.parquet", "data/polymath/hi/top.parquet"),
        ),
    ),
    "vmlu": ("tridm/VMLU", (("test.jsonl", "data/vmlu/test.jsonl"),)),
    "m_arena_hard_vi": (
        "CohereLabs/m_ArenaHard",
        (("vi/test-00000-of-00001.parquet", "data/m_arena_hard/vi/test-00000-of-00001.parquet"),),
    ),
    "m_arena_hard_hi": (
        "CohereLabs/m_ArenaHard",
        (("hi/test-00000-of-00001.parquet", "data/m_arena_hard/hi/test-00000-of-00001.parquet"),),
    ),
    "milu_1_5": (
        "ai4bharat/MILU-1.5",
        (("Hindi/test-00000-of-00001.parquet", "data/milu_1_5/Hindi/test-00000-of-00001.parquet"),),
    ),
    "ifeval": ("google/IFEval", (("ifeval_input_data.jsonl", "data/ifeval/ifeval_input_data.jsonl"),)),
}

_GSM8K_EN_GITHUB = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
)
_IFEVAL_GITHUB = (
    "https://raw.githubusercontent.com/google-research/google-research/master/"
    "instruction_following_eval/data/input_data.jsonl"
)


def _token() -> str | bool:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or False


def _copy_hub_file(repo: str, remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=remote, repo_type="dataset", token=_token())
    shutil.copy2(path, dest)
    LOGGER.info("fetched %s -> %s", remote, dest)


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    LOGGER.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as handle:
        shutil.copyfileobj(resp, handle)


def fetch_source(source: str, data_root: Path) -> None:
    if source in {"tiny"}:
        return
    root = data_root.resolve()
    # Hub paths are written as data/... ; remap onto data_root.
    spec = HF_DATASETS.get(source)
    if spec is None:
        LOGGER.warning("no Hugging Face mapping for source %s", source)
        return
    repo, files = spec
    try:
        for remote, local in files:
            dest = root / Path(local).relative_to("data")
            _copy_hub_file(repo, remote, dest)
    except Exception as exc:  # noqa: BLE001
        if source == "gsm8k_en":
            _download_url(_GSM8K_EN_GITHUB, root / "gsm8k_en" / "test.jsonl")
            return
        if source == "ifeval":
            _download_url(_IFEVAL_GITHUB, root / "ifeval" / "ifeval_input_data.jsonl")
            return
        raise RuntimeError(
            f"could not fetch {source} from {repo}: {exc}. "
            "Set HF_TOKEN for gated sets (MILU-1.5) or place files under data_root."
        ) from exc


def fetch_sources(sources: tuple[str, ...], data_root: Path) -> None:
    extra: list[str] = []
    for source in sources:
        extra.append(source)
        if source.startswith("mmlu_prox_") and not source.endswith("_val"):
            extra.append(f"{source}_val")
    seen: set[str] = set()
    for source in extra:
        if source in seen:
            continue
        seen.add(source)
        fetch_source(source, data_root)

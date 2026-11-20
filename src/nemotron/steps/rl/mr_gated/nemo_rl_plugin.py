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

"""NeMo-RL extensions that make the hard-gate reward the GRPO verifier.

Listed under ``env.plugins`` in an ``rl/nemo_rl/rlvr`` config, this module registers:

- dataset ``mr_gated_gsm8k``: gated prompt rows from ``data.py``
- processor ``mr_gated_processor``: carries each gated row to the environment
- environment ``mr_gated``: ``gated_reward`` on every rollout

Imports NeMo-RL, so it only loads inside the NeMo-RL image.
"""

from __future__ import annotations

import collections
from typing import Any

import ray
import torch
from datasets import Dataset
from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
from nemo_rl.data.processors import PROCESSOR_REGISTRY, math_hf_data_processor, register_processor
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.environments.math_environment import BaseMathEnvironment
from nemo_rl.environments.utils import ENV_REGISTRY, register_env
from transformers import AutoTokenizer

from nemotron.steps.eval.mr_l2.parser import answers_match
from nemotron.steps.rl.mr_gated.data import build_rows
from nemotron.steps.rl.mr_gated.reward import gated_reward
from nemotron.steps.rl.mr_gated.schema import GATED_FIELDS

DATASET_NAME = "mr_gated_gsm8k"
PROCESSOR_NAME = "mr_gated_processor"
ENV_NAME = "mr_gated"


class MrGatedGsm8kDataset(RawDataset):
    """Gated prompt rows; the prompt is the plain question and the assistant turn holds the label."""

    def __init__(self, split: str = "train", english_fraction: float = 0.25, seed: int = 42, **kwargs) -> None:
        self.task_name = DATASET_NAME
        self.dataset = Dataset.from_list(
            [
                {
                    **row,
                    "messages": [
                        {"role": "user", "content": row["unassisted_user"]},
                        {"role": "assistant", "content": row["label"]},
                    ],
                    "task_name": self.task_name,
                }
                for row in build_rows(split, english_fraction, seed)
            ]
        )
        self.val_dataset = None


def mr_gated_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    """``math_hf_data_processor`` with the gated row as the environment's metadata."""
    output = math_hf_data_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx)
    output["extra_env_info"] = {key: datum_dict[key] for key in GATED_FIELDS}
    return output


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class MrGatedEnvironment(BaseMathEnvironment):
    """Scores each rollout with ``gated_reward``: +1, 0 or -1.

    Config (``env.mr_gated``): ``tokenizer_name`` and ``max_tokens`` give the
    reward the token count and budget its closure check expects; ``log_every``
    sets how often gate rates per language are printed.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.workers = []  # the reward is computed in this actor
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
        self.stats: collections.Counter = collections.Counter()

    def step(self, message_log_batch, metadata, return_extracted_answer: bool = False) -> EnvironmentReturn:
        responses = [
            "".join(str(message["content"]) for message in log if message["role"] == "assistant")
            for log in message_log_batch
        ]
        token_counts = [len(ids) for ids in self.tokenizer(responses, add_special_tokens=False)["input_ids"]]
        results = [
            gated_reward(response, row, n_tokens=n_tokens, max_tokens=self.cfg.get("max_tokens"))
            for response, row, n_tokens in zip(responses, metadata, token_counts)
        ]
        self._log(results, metadata)
        rewards = [float(result["reward"]) for result in results]
        return EnvironmentReturn(
            observations=[
                {"role": "environment", "content": "Environment: " + ("correct" if value > 0 else "incorrect")}
                for value in rewards
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32).cpu(),
            terminateds=torch.ones(len(rewards)).cpu(),
            answers=[result["parsed"] or None for result in results] if return_extracted_answer else None,
        )

    def _log(self, results, metadata) -> None:
        for result, row in zip(results, metadata):
            lang = row["prompt_language"]
            self.stats[lang, "n"] += 1
            self.stats[lang, {1: "plus", 0: "zero", -1: "minus"}[result["reward"]]] += 1
            self.stats[lang, "label_match"] += answers_match(result["parsed"], row["label"], "numeric")
            for key in ("thinking_lang_ok", "answer_lang_ok", "closed", "truncated", "loop"):
                self.stats[lang, key] += bool(result[key])
        if sum(count for (_, key), count in self.stats.items() if key == "n") < self.cfg.get("log_every", 448):
            return
        for lang in sorted({lang for lang, _ in self.stats}):
            n = self.stats[lang, "n"]
            rates = " ".join(
                f"{key}={self.stats[lang, key] / n:.3f}"
                for key in (
                    "plus",
                    "zero",
                    "minus",
                    "label_match",
                    "thinking_lang_ok",
                    "answer_lang_ok",
                    "closed",
                    "truncated",
                    "loop",
                )
            )
            print(f"[mr_gated] lang={lang} n={n} {rates}", flush=True)
        self.stats.clear()


def register() -> None:
    """Add the dataset, processor and environment to NeMo-RL's registries. Safe to call twice."""
    actor = f"{__name__}.MrGatedEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY.setdefault(actor, PY_EXECUTABLES.SYSTEM)
    if ENV_NAME not in ENV_REGISTRY:
        register_env(ENV_NAME, actor)
    DATASET_REGISTRY.setdefault(DATASET_NAME, MrGatedGsm8kDataset)
    if PROCESSOR_NAME not in PROCESSOR_REGISTRY:
        register_processor(PROCESSOR_NAME, mr_gated_processor)

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

"""Pinned harness revisions for the Multilingual Reasoning agent benches."""

from __future__ import annotations

SEATAU_GIT = "https://github.com/SEACrowd/SEATauBench.git"
SEATAU_SHA = "312f3e6eb87c94ad2554a74cc4f4b96baa6f89f9"
TAU2_GIT = "https://github.com/sierra-research/tau2-bench.git"
TAU2_SHA = "033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

SEATAU_DOMAINS = ("airline", "retail", "telecom")
SEATAU_SCENARIOS = ("english", "l2_tools", "l2_interaction", "l2_domain")
# Sheet Multilingual Reasoning: HI/EN headline is all five domains, not text-v1 50×3.
TAU2_DOMAINS = ("airline", "retail", "telecom", "banking_knowledge", "intake")
CS_OVERLAP_DOMAINS = ("airline", "retail", "telecom")

DEFAULT_USER_MODEL = "openai/azure/openai/gpt-5.5"
DEFAULT_USER_API_BASE = "https://inference-api.nvidia.com/v1"
DEFAULT_SEATAU_SCENARIO = "l2_interaction"
DEFAULT_SEATAU_TRIALS = 3
DEFAULT_TAU2_TRIALS = 4
INTAKE_ELICITATION_STRATEGY = "scaffolded"

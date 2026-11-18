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


def hard_gate(
    *,
    thinking_lang_ok: bool,
    answer_lang_ok: bool,
    closed: bool,
    answer_matches_label: bool,
) -> int:
    """
    Frozen 1 / 0 / −1 rule.

    Language fail always −1. Correctness and closure only distinguish 1 vs 0
    after both language conditions pass.

    Unlabeled items (<6/8) must be excluded by the caller, not scored 0.
    """
    if not thinking_lang_ok or not answer_lang_ok:
        return -1
    if answer_matches_label and closed:
        return 1
    return 0

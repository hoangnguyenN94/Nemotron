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

"""Native-reasoning hard-gate reward for gated RL ({+1, 0, −1})."""

from nemotron.steps.rl.mr_gated.reward import gated_reward
from nemotron.steps.rl.mr_gated.runtime import run_mr_gated
from nemotron.steps.rl.mr_gated.schema import TrainSchemaError, refuse_gold, validate_gated_record

__all__ = [
    "TrainSchemaError",
    "gated_reward",
    "refuse_gold",
    "run_mr_gated",
    "validate_gated_record",
]

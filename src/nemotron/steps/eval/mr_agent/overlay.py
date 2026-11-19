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

"""Apply Native-reasoning think-prefix and NVIDIA user-sim overlays onto a clone."""

from __future__ import annotations

import shutil
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent
OVERLAY_DIR = STEP_DIR / "overlay"

THINKING_IMPORT = """from tau2.agent.thinking_overlay import (
    fold_reasoning,
    prepare_thinking_generate,
    stitch_thinking_prefill,
)
"""

# First `_generate_next_message` on LLMAgent in the pinned SHAs.
UPSTREAM_GENERATE = """        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message
"""

THINKING_GENERATE = """        prefill, llm_args = prepare_thinking_generate(self.llm_args)
        messages = state.system_messages + state.messages
        if prefill is not None:
            messages = list(messages) + [prefill]
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **llm_args,
        )
        assistant_message = fold_reasoning(assistant_message)
        return stitch_thinking_prefill(assistant_message, prefill)
"""


def apply_thinking_overlay(llm_agent_path: Path, *, thinking_src: Path | None = None) -> None:
    """Inject forced-think helpers into cloned `src/tau2/agent/llm_agent.py`."""
    dest_thinking = llm_agent_path.parent / "thinking_overlay.py"
    src = thinking_src or (OVERLAY_DIR / "thinking.py")
    shutil.copy2(src, dest_thinking)
    text = llm_agent_path.read_text(encoding="utf-8")
    if "prepare_thinking_generate" in text:
        return
    if "from tau2.utils.llm_utils import generate" not in text:
        raise RuntimeError(f"{llm_agent_path} is missing the generate import; SHA may have drifted")
    if UPSTREAM_GENERATE not in text:
        raise RuntimeError(
            f"{llm_agent_path} LLMAgent._generate_next_message does not match the pinned SHA. "
            "Refresh eval/mr_agent overlay.py against the new tau2/SEATau revision."
        )
    text = text.replace(
        "from tau2.utils.llm_utils import generate\n",
        "from tau2.utils.llm_utils import generate\n" + THINKING_IMPORT,
        1,
    )
    text = text.replace(UPSTREAM_GENERATE, THINKING_GENERATE, 1)
    llm_agent_path.write_text(text, encoding="utf-8")


def apply_config_overlay(clone_root: Path, *, harness: str) -> None:
    """Overwrite `src/tau2/config.py` with the NVIDIA inference-api routing overlay."""
    src = OVERLAY_DIR / harness / "config.py"
    dest = clone_root / "src" / "tau2" / "config.py"
    if not src.is_file():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def apply_harness_overlay(clone_root: Path, *, harness: str) -> None:
    apply_thinking_overlay(clone_root / "src" / "tau2" / "agent" / "llm_agent.py")
    apply_config_overlay(clone_root, harness=harness)

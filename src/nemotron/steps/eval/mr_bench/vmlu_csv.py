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

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from nemotron.steps.eval.mr_bench.ids import parse_item_id


def native_vmlu_id(item_id: str) -> str:
    source, native = parse_item_id(item_id)
    if source != "vmlu":
        raise ValueError(f"not a VMLU id: {item_id}")
    return native


def write_vmlu_submission_csv(records: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    """Write the vmlu.ai leaderboard CSV: header `id,answer` then one row per item."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "answer"])
        for rec in rows:
            item_id = str(rec.get("item_id") or rec.get("id") or "")
            native = rec.get("native_id")
            if native:
                vid = str(native)
            elif item_id.startswith("vmlu:"):
                vid = native_vmlu_id(item_id)
            else:
                vid = item_id
            answer = rec.get("parsed_answer")
            if answer is None:
                answer = rec.get("answer") or rec.get("predicted_answer") or ""
            writer.writerow([vid, str(answer)])
    return out

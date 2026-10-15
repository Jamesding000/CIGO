# Copyright 2026 The CIGO Authors
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
"""One reward_extra_info schema for a batch that mixes reward functions.

Rows of different datasets are scored by different functions (multiple choice vs IoU), so their
reward_extra_info keys differ, per row and per worker chunk. Missing keys are padded with NaN so every row
and chunk carries the same columns, in first-seen order.
"""

import numpy as np

from verl.protocol import DataProto


def align_reward_extra_infos(reward_extra_infos: list[dict]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Per-row reward_extra_info dicts -> ({key: object array over rows}, union of keys)."""
    union_keys: list[str] = list({k: None for info in reward_extra_infos for k in info})
    aligned: dict[str, np.ndarray] = {}
    for key in union_keys:
        arr = np.empty(len(reward_extra_infos), dtype=object)
        arr[:] = [info.get(key, np.nan) for info in reward_extra_infos]
        aligned[key] = arr
    return aligned, union_keys


def align_non_tensor_batches_across_chunks(outputs: list[DataProto], key_meta_field: str = "reward_extra_keys") -> None:
    """Give every worker chunk the union of the chunks' reward_extra_keys, in place: DataProto.concat
    requires identical keys."""
    union_keys: list[str] = list({k: None for out in outputs for k in out.meta_info.get(key_meta_field, [])})
    if not union_keys:
        return
    for out in outputs:
        chunk_size = len(out)
        for key in union_keys:
            if key not in out.non_tensor_batch:
                arr = np.empty(chunk_size, dtype=object)
                arr[:] = np.nan
                out.non_tensor_batch[key] = arr
        out.meta_info[key_meta_field] = union_keys

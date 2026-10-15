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
"""Extra per-row columns of the rollout dump and the task-reward metrics."""

import numpy as np
import torch

ROLLOUT_DUMP_COLUMNS = (
    "uid",
    "n_truncated_turns",
    "prefix_action_type",
    "prefix_depth",
    "prefix_entry_id",
    "turn_ur",
    "think_ur",
    "u_prompt",
    "u_after_think0",
    "u_terminal",
    "tool_exec_error",
    "partial_format",
    "over_budget",
    "pure_format",
    "budget_declared",
    "remaining_fixed",
    "pr_payoff",
    "pr_adv_payoff",
    "pr_adv_ur",
    "pr_adv",
    "pr_selected",
)


def json_default(o):
    """json.dumps fallback for the numpy and torch values the dump columns carry."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def add_rollout_dump_columns(columns: dict, non_tensor_batch: dict):
    """Add the ROLLOUT_DUMP_COLUMNS the batch carries to the dump's per-row columns."""
    for col in ROLLOUT_DUMP_COLUMNS:
        if col in non_tensor_batch:
            columns.setdefault(col, list(non_tensor_batch[col]))


def task_reward_metrics(non_tensor_batch: dict) -> dict:
    """critic/rewards/<key>/mean of the task reward (0/1 multiple choice, IoU grounding) and of the strict
    format verdict, ignoring the NaN of rows whose scorer does not emit the key."""
    metrics = {}
    for key in ("mcq_reward", "iou_reward", "pure_format"):
        if key in non_tensor_batch:
            vals = np.asarray(non_tensor_batch[key], dtype=np.float32)
            if not np.all(np.isnan(vals)):
                metrics[f"critic/rewards/{key}/mean"] = float(np.nanmean(vals))
    return metrics

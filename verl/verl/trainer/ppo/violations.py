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
"""The rules a rollout is held to and their price, shared by the GRPO and prefix-group estimators.

A row that breaks any rule (judged on its own turn) gets raw reward -algorithm.violation_penalty, below
the 0 of a legal wrong answer; breaking several rules is charged once.
"""

from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig


def violation_penalty(config) -> float:
    """``algorithm.violation_penalty``, the price of every rule (> 0, so a violating row ranks below a legal
    wrong answer)."""
    if config is None:
        return AlgoConfig.violation_penalty
    penalty = float(config.get("violation_penalty", AlgoConfig.violation_penalty))
    assert penalty > 0.0, f"algorithm.violation_penalty must be > 0, got {penalty}"
    return penalty


def prefix_violation(i: int, non_tensor_batch: dict) -> Optional[str]:
    """The first rule row ``i`` breaks on its own turn, else None.

    In order: over_budget (called with no calls left), truncated (hit the per-turn token cap),
    tool_exec_error (a well-formed call the tool could not run), bad_format (rejected by ``pure_format``,
    including a row that took no action). The order only decides the name reported under ``pr/violation/*``.
    """
    col = non_tensor_batch.get("over_budget")
    if col is not None and bool(col[i]):
        return "over_budget"
    trunc = non_tensor_batch.get("n_truncated_turns")
    if trunc is not None and trunc[i] is not None and float(trunc[i]) > 0:
        return "truncated"
    err = non_tensor_batch.get("tool_exec_error")
    if err is not None and err[i] is not None and bool(err[i]):
        return "tool_exec_error"
    # NaN is not a pass
    pf = non_tensor_batch.get("pure_format")
    v = pf[i] if pf is not None else None
    if not (v is not None and float(v) == float(v) and float(v) > 0.5):
        return "bad_format"
    return None


def violation_rules(non_tensor_batch: dict) -> np.ndarray:
    """``prefix_violation`` of every row as an object array (None for a legal row)."""
    if non_tensor_batch.get("pure_format") is None:
        raise ValueError("non_tensor_batch has no 'pure_format' column — the strict format scorer is not wired.")
    rules = np.empty(len(non_tensor_batch["pure_format"]), dtype=object)
    for i in range(len(rules)):
        rules[i] = prefix_violation(i, non_tensor_batch)
    return rules


def price_violations(
    scores: torch.Tensor,
    non_tensor_batch: Optional[dict],
    config: Optional[AlgoConfig],
) -> torch.Tensor:
    """Violation pricing for GRPO: r_i <- -algorithm.violation_penalty on every row that breaks a rule, as in
    ``compute_prefix_group_advantage``. Records the rules in ``non_tensor_batch["violation"]``; without a
    ``non_tensor_batch`` the scores are returned unchanged.
    """
    if non_tensor_batch is None:
        return scores
    rules = violation_rules(non_tensor_batch)
    non_tensor_batch["violation"] = rules
    penalty = violation_penalty(config)
    with torch.no_grad():
        for i, rule in enumerate(rules):
            if rule is not None:
                scores[i] = -penalty
    return scores

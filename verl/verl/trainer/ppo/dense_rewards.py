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
"""The GRPO arms' dense term, added to the outcome advantages after compute_advantage."""

import numpy as np
import torch

from verl import DataProto
from verl.trainer.config import AlgoConfig


def add_dense_advantages(data: DataProto, tokenizer, config: AlgoConfig) -> dict:
    """A <- A + coef * A_dense, on advantages and returns, where A_dense is the intrinsic (process) reward
    advantage (GRPO + IR) or the annotation reward advantage (GRPO + AR), set to 0 on rows that broke a rule.
    Returns the term's metrics.
    """
    from verl.trainer.ppo import turn_uncertainty

    if config.process_reward.enable:
        A, metrics = turn_uncertainty.compute_turn_uncertainty_advantage(data, tokenizer, config.probe_model_path)
        coef, name = float(config.process_reward.coef), "process_advantages"
    elif config.annotation_reward.enable:
        A, metrics = turn_uncertainty.compute_annotation_segment_advantage(data, tokenizer)
        coef, name = float(config.annotation_reward.coef), "annotation_advantages"
    else:
        return {}
    A = A.to(data.batch["advantages"].device, dtype=data.batch["advantages"].dtype)
    penalised = np.array([rule is not None for rule in data.non_tensor_batch["violation"]])
    if penalised.any():
        A[torch.as_tensor(penalised, device=A.device)] = 0.0
    contribution = coef * A
    data.batch["advantages"] = data.batch["advantages"] + contribution
    data.batch["returns"] = data.batch["returns"] + contribution
    valid = torch.masked_select(A, data.batch["response_mask"].bool())
    metrics[f"critic/{name}/abs_mean"] = valid.abs().mean().item()
    return metrics

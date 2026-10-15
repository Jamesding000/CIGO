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
"""Start-up steps of main_ppo: the scorer's environment for the Ray actors and a copy of the resolved config."""

import os

from omegaconf import OmegaConf


def forward_scorer_env(runtime_env_kwargs):
    """Forward LOGGER_OUTPUT_FILE (the reward scorers' log file) to the Ray actors' env_vars."""
    forwarded = {k: os.environ[k] for k in ("LOGGER_OUTPUT_FILE",) if k in os.environ}
    if forwarded:
        # a new config: the composed one is in struct mode
        runtime_env_kwargs = OmegaConf.merge({}, runtime_env_kwargs, {"env_vars": forwarded})
        print(f"[runtime_env] forwarding scoring knobs to Ray actors: {forwarded}")
    return runtime_env_kwargs


def save_resolved_config(config):
    """Write the resolved config to trainer.default_local_dir/resolved_config.yaml before training starts."""
    local_dir = config.trainer.get("default_local_dir", None)
    if local_dir:
        try:
            os.makedirs(local_dir, exist_ok=True)
            OmegaConf.save(config, os.path.join(local_dir, "resolved_config.yaml"), resolve=True)
            print(f"resolved config written to {local_dir}/resolved_config.yaml")
        except OSError as e:
            print(f"could not write resolved_config.yaml: {e}")

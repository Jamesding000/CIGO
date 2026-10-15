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
"""Sampler that shuffles consecutive pairs of rows.

Rows 2i and 2i+1 stay adjacent; the order of the pairs is a permutation drawn from a generator
seeded with `data.seed` (1 when null); an odd last row is dropped.
"""

from collections.abc import Sized

import torch
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class PairShuffleSampler(AbstractCurriculumSampler):
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(1 if data_config.get("seed") is None else int(data_config.seed))
        self.data_source = data_source
        self.generator = train_dataloader_generator
        self.T = 2

    def __iter__(self):
        n = len(self.data_source) // self.T
        perm = torch.randperm(n, generator=self.generator).tolist()
        for idx in perm:
            for i in range(self.T):
                yield idx*self.T + i

    def __len__(self) -> int:
        return len(self.data_source) // self.T * self.T

    def update(self, batch) -> None:
        return


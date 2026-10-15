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
"""Driver-side replay buffer of trajectory prefixes for prefix replay, saved as JSON with each checkpoint.

An entry is a prefix of a tool-use trajectory: its token ids before video expansion plus the tool-call
arguments needed to re-fetch the frames.
"""

import json
import logging
import os
import uuid
from collections import Counter
from typing import Any, Optional

import numpy as np

from verl.trainer.config import PrefixReplayConfig
from verl.utils.config import omega_conf_to_dataclass

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ROW_KEYS = ("raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "index")


def _to_jsonable(obj: Any) -> Any:
    """OmegaConf containers and numpy scalars as plain Python values."""
    if isinstance(obj, dict) or type(obj).__name__ == "DictConfig":
        return {str(k): _to_jsonable(v) for k, v in dict(obj).items()}
    if isinstance(obj, (list, tuple)) or type(obj).__name__ == "ListConfig":
        return [_to_jsonable(v) for v in list(obj)]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class PrefixReplayBuffer:
    """Mid-trajectory prefixes; sampled entries become training rows.

    An entry holds entry_id (the group uid when sampled), row (the root example's ROW_KEYS), prefix_ids
    (token ids before video expansion, ending after a tool result), tool_calls (executed
    {"name", "arguments"} in order), user_turns / assistant_turns, depth (tool turns in the prefix),
    expanded_len (length after video expansion), partial_format, created_step and visits.
    """

    def __init__(self, config, prompt_length: Optional[int] = None):
        cfg = config if isinstance(config, PrefixReplayConfig) else omega_conf_to_dataclass(config, PrefixReplayConfig)
        self.capacity = int(cfg.capacity)
        self.ttl_steps = int(cfg.ttl_steps)
        self.max_visits = int(cfg.max_visits)
        self.max_depth = int(cfg.max_depth)
        self.recency_gamma = float(cfg.recency_gamma)
        self.admission_margin = int(cfg.admission_margin)
        self.max_source_share = float(cfg.max_source_share)
        self.source_share_min_size = int(cfg.source_share_min_size)
        self.max_inserts_per_step = int(cfg.max_inserts_per_step)
        self.prompt_length = prompt_length

        self.entries: dict[str, dict] = {}
        self._dedup_keys: set = set()
        self._sampled_age_mean: Optional[float] = None  # of the last sample() draw

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _source_key(row: dict) -> str:
        """The seed corpus of a row (extra_info.corpus, else data_source), for the share cap."""
        extra = row.get("extra_info") or {}
        return str(extra.get("corpus") or row.get("data_source") or "unknown")

    def _source_counts(self) -> Counter:
        c = Counter()
        for e in self.entries.values():
            c[self._source_key(e["row"])] += 1
        return c

    def _dedup_key(self, cont: dict):
        calls = tuple((c["name"], c["arguments"]) for c in cont.get("tool_calls", []))
        return (cont["row"].get("index"), cont.get("depth"), calls)

    def add(self, cont: dict, global_step: int, parent_live: Optional[bool] = None) -> bool:
        """Buffer admission: insert a continuation iff depth <= max_depth, expanded_len <= prompt_length -
        admission_margin, partial_format passes, not over budget, its parent group is live (raw rewards
        differ; None is not gated), it is not a duplicate, and its corpus stays within max_source_share of
        the buffer (enforced once the buffer holds source_share_min_size entries). Evicts the oldest entry
        beyond capacity. Returns True if inserted.
        """
        if cont.get("depth", 0) > self.max_depth:
            return False
        if self.prompt_length is not None and cont.get("expanded_len", 0) > self.prompt_length - self.admission_margin:
            return False
        # mid-trace format rule (cigo/reward/format_gate.py), not pure_format, which demands an <answer>
        pf = cont.get("partial_format")
        if pf is None or float(pf) <= 0.5:
            return False
        if bool(cont.get("over_budget", False)):
            return False
        if parent_live is False:
            return False

        key = self._dedup_key(cont)
        if key in self._dedup_keys:
            return False

        if self.max_source_share < 1.0 and len(self.entries) >= self.source_share_min_size:
            src = self._source_key(cont["row"])
            allowed = int(self.max_source_share * (len(self.entries) + 1))
            if self._source_counts()[src] + 1 > allowed:
                return False

        entry = {
            "entry_id": uuid.uuid4().hex,
            "row": _to_jsonable({k: cont["row"].get(k) for k in ROW_KEYS}),
            "prefix_ids": [int(t) for t in cont["prefix_ids"]],
            "tool_calls": _to_jsonable(cont.get("tool_calls", [])),
            "user_turns": int(cont.get("user_turns", 0)),
            "assistant_turns": int(cont.get("assistant_turns", 0)),
            "depth": int(cont.get("depth", 1)),
            "expanded_len": int(cont.get("expanded_len", len(cont["prefix_ids"]))),
            "partial_format": None if cont.get("partial_format") is None else float(cont["partial_format"]),
            "created_step": int(global_step),
            "visits": 0,
        }
        self.entries[entry["entry_id"]] = entry
        self._dedup_keys.add(key)

        if len(self.entries) > self.capacity:
            oldest = min(self.entries.values(), key=lambda e: e["created_step"])
            self._remove(oldest["entry_id"])
        return True

    def _remove(self, entry_id: str):
        e = self.entries.pop(entry_id, None)
        if e is not None:
            self._dedup_keys.discard(self._dedup_key(e))

    def remove(self, entry_id: str):
        """Evict one entry."""
        self._remove(entry_id)

    def _evict_stale(self, global_step: int):
        stale = [
            eid
            for eid, e in self.entries.items()
            if (global_step - e["created_step"] > self.ttl_steps) or (e["visits"] >= self.max_visits)
        ]
        for eid in stale:
            self._remove(eid)

    def sample(self, k: int, global_step: int) -> list[dict]:
        """Draw up to k entries without replacement with P(e) proportional to recency_gamma ** age(e), after
        evicting entries older than ttl_steps or visited max_visits times."""
        self._sampled_age_mean = None
        self._evict_stale(global_step)
        if k <= 0 or not self.entries:
            return []
        pool = list(self.entries.values())
        k = min(k, len(pool))
        ages = np.array([global_step - e["created_step"] for e in pool], dtype=np.float64)
        w = np.power(self.recency_gamma, ages)
        p = w / w.sum()
        idx = np.random.choice(len(pool), size=k, replace=False, p=p)
        sampled = [pool[i] for i in idx]
        for e in sampled:
            e["visits"] += 1
        self._sampled_age_mean = float(np.mean([global_step - e["created_step"] for e in sampled]))
        return sampled

    def metrics(self) -> dict:
        """Buffer size, mean prefix depth, mean age of the sampled entries and the largest corpus share."""
        m = {"prefix_buffer/size": float(len(self.entries))}
        if self.entries:
            m["prefix_buffer/source_share_max"] = max(self._source_counts().values()) / len(self.entries)
            m["prefix_buffer/depth_mean"] = float(np.mean([e["depth"] for e in self.entries.values()]))
        if self._sampled_age_mean is not None:
            m["prefix_buffer/sampled_age_mean"] = self._sampled_age_mean
        return m

    def save_to_file(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # atomic write
        tmp = filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(list(self.entries.values()), f)
        os.replace(tmp, filepath)
        logger.info(f"Saved prefix replay buffer ({len(self.entries)} entries) to {filepath}")

    def load_from_file(self, data_path: str):
        with open(data_path) as f:
            raw = json.load(f)
        for e in raw:
            self.entries[e["entry_id"]] = e
            self._dedup_keys.add(self._dedup_key(e))
        logger.info(f"Loaded prefix replay buffer ({len(self.entries)} entries) from {data_path}")

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
"""Prefix replay on the trainer side: the hooks RayPPOTrainer.fit() calls at each stage of a step.

The driver assembles each batch from fresh dataloader rows and buffer prefixes, pushes continuations
back into the buffer, runs the uncertainty probe before the advantage and logs the advantage metrics.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import violations


def loader_batch_size(config) -> int:
    """Rows the train dataloader yields per step: the whole batch, or under prefix replay only its fresh
    share round(train_batch_size * fresh_root_ratio) (the rest are buffer prefixes)."""
    batch_prompts = _batch_prompts(config)
    prefix_replay_cfg = config.algorithm.prefix_replay
    if not bool(prefix_replay_cfg.enable):
        return batch_prompts
    fresh_prompts = int(round(batch_prompts * float(prefix_replay_cfg.fresh_root_ratio)))
    assert 1 <= fresh_prompts <= batch_prompts, (
        f"algorithm.prefix_replay.fresh_root_ratio leaves {fresh_prompts} fresh prompts of "
        f"{batch_prompts}; it must leave at least 1 and at most the whole batch"
    )
    return fresh_prompts


def _batch_prompts(config) -> int:
    return int(config.data.get("gen_batch_size", config.data.train_batch_size))


def save_loader_meta(step_folder: str, batch_size: int):
    """Record the dataloader batch size next to its saved position (data.pt), which is counted in batches."""
    with open(os.path.join(step_folder, "data_meta.json"), "w") as f:
        json.dump({"train_loader_batch_size": int(batch_size)}, f)


def check_loader_meta(step_folder: str, batch_size: int):
    """Refuse to restore a dataloader position saved at a different batch size; warn when none was recorded."""
    meta_path = os.path.join(step_folder, "data_meta.json")
    saved_loader_batch_size = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            saved_loader_batch_size = json.load(f).get("train_loader_batch_size")
    if saved_loader_batch_size is None:
        print(
            f"Warning: {meta_path} is absent. Assuming its "
            f"dataloader batch size was the current "
            f"{batch_size}; if it was not, the restored position "
            f"points at a different row."
        )
    elif int(saved_loader_batch_size) != int(batch_size):
        raise ValueError(
            f"dataloader batch size is {batch_size} but this "
            f"checkpoint was written at {saved_loader_batch_size}, so its saved "
            f"position means a different number of consumed rows. Resume with the "
            f"data.train_batch_size and algorithm.prefix_replay.fresh_root_ratio the "
            f"run started on, or start a new trainer.experiment_name."
        )


def build_driver(trainer) -> "NoPrefixReplay":
    """The prefix-replay hooks of this run (no-ops without prefix replay). Call after the checkpoint is
    loaded, so the driver can restore the buffer saved with it."""
    if bool(trainer.config.algorithm.prefix_replay.enable):
        return PrefixReplayDriver(trainer)
    return NoPrefixReplay(trainer.config)


class NoPrefixReplay:
    """The hooks of a run without prefix replay: each is a no-op."""

    def __init__(self, config):
        self.config = config

    def assemble(self, batch_dict: dict, global_steps: int) -> dict:
        return batch_dict

    def after_reward(self, batch: DataProto, metrics: dict, global_steps: int):
        pass

    def before_advantage(self, batch: DataProto, metrics: dict):
        pass

    def after_advantage(self, batch: DataProto, metrics: dict):
        pass

    def save(self, global_steps: int):
        pass


class PrefixReplayDriver(NoPrefixReplay):
    """Prefix replay on the driver: the prefix buffer, batch assembly, continuation push-back, the
    uncertainty probe and the advantage metrics."""

    def __init__(self, trainer):
        super().__init__(trainer.config)
        config = trainer.config
        replay_cfg = config.algorithm.prefix_replay
        assert str(config.algorithm.adv_estimator) == "prefix_group", (
            "algorithm.prefix_replay.enable=True requires algorithm.adv_estimator=prefix_group, "
            f"got {config.algorithm.adv_estimator}"
        )
        assert not config.algorithm.use_kl_in_reward, (
            "prefix_replay is incompatible with algorithm.use_kl_in_reward; use actor.use_kl_loss"
        )
        assert config.algorithm.probe_model_path, (
            "algorithm.prefix_replay.enable requires algorithm.probe_model_path (the frozen probe model)"
        )
        from verl.trainer.ppo.prefix_replay_buffer import PrefixReplayBuffer

        self.tokenizer = trainer.tokenizer
        self.total_training_steps = trainer.total_training_steps
        self.batch_prompts = _batch_prompts(config)
        self.buffer = PrefixReplayBuffer(replay_cfg, prompt_length=config.actor_rollout_ref.rollout.prompt_length)
        # the checkpointed step is global_steps - 1
        resumed_step = trainer.global_steps - 1
        if resumed_step > 0:
            ckpt_buffer = self._buffer_path(resumed_step)
            if os.path.exists(ckpt_buffer):
                self.buffer.load_from_file(ckpt_buffer)
            else:
                print(
                    f"[prefix_replay] WARNING: resuming from step {resumed_step} but no "
                    f"{ckpt_buffer} found — buffer cold-starts empty"
                )
        print(
            f"[prefix_replay] enabled: buffer size={len(self.buffer)}, "
            f"fresh_root_ratio={float(replay_cfg.fresh_root_ratio)}"
        )
        print(
            f"[prefix_replay] batch={self.batch_prompts} prompts = "
            f"{trainer.train_loader_batch_size} fresh + "
            f"{self.batch_prompts - trainer.train_loader_batch_size} replayed; "
            f"steps per epoch={len(trainer.train_dataloader)}, "
            f"total_training_steps={self.total_training_steps}"
        )

    def _buffer_path(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"prefix_buffer_step_{step}.json")

    def assemble(self, batch_dict: dict, global_steps: int) -> dict:
        """Append buffer-sampled prefix rows to the fresh batch and set agent_name / prefix_state on every
        row (fresh rows: prefix_state None, rolled out from the root).

        Appended rows copy a fresh row's tensor columns as placeholders; the agent loop rebuilds their input
        from raw_prompt and prefix_ids. A buffer shortfall is filled with duplicated fresh rows, keeping the
        batch size fixed.
        """
        n_fresh = len(batch_dict["data_source"])
        n_replay = max(0, self.batch_prompts - n_fresh)
        entries = self.buffer.sample(n_replay, global_steps) if n_replay else []

        bsz = n_fresh + n_replay
        if n_replay:
            # wrap around: the replayed share may exceed the fresh one
            take = np.arange(n_replay) % n_fresh
            for key, val in list(batch_dict.items()):
                if isinstance(val, torch.Tensor):
                    batch_dict[key] = torch.cat([val, val[torch.as_tensor(take, dtype=torch.long)]], dim=0)
                elif isinstance(val, np.ndarray):
                    batch_dict[key] = np.concatenate([val, val[take]])
                elif isinstance(val, list):
                    batch_dict[key] = val + [val[i] for i in take]
                else:
                    raise TypeError(
                        f"PrefixReplayDriver.assemble cannot extend column {key!r} of type "
                        f"{type(val).__name__}; a column left at {n_fresh} rows while the rest "
                        f"reach {bsz} produces a ragged batch that DataProto will not catch"
                    )
            ragged = {k: len(v) for k, v in batch_dict.items() if len(v) != bsz}
            assert not ragged, f"PrefixReplayDriver.assemble produced a ragged batch: {ragged} != {bsz}"
        agent_names = np.empty(bsz, dtype=object)
        agent_names[:] = "prefix_replay_agent"
        prefix_states = np.empty(bsz, dtype=object)
        prefix_states[:] = None

        for j, entry in enumerate(entries):
            i = n_fresh + j
            for key in ("raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "index"):
                if key in batch_dict:
                    batch_dict[key][i] = entry["row"][key]
            prefix_states[i] = {
                "entry_id": entry["entry_id"],
                "prefix_ids": entry["prefix_ids"],
                "tool_calls": entry["tool_calls"],
                "user_turns": entry["user_turns"],
                "assistant_turns": entry["assistant_turns"],
                "depth": entry["depth"],
            }
        batch_dict["agent_name"] = agent_names
        batch_dict["prefix_state"] = prefix_states
        return batch_dict

    def after_reward(self, batch: DataProto, metrics: dict, global_steps: int):
        """Insert the batch's continuations into the buffer (``PrefixReplayBuffer.add``) and log behaviour and
        buffer metrics. A group is live when its raw rewards differ."""
        ntb = batch.non_tensor_batch
        action_type = ntb["prefix_action_type"]
        uid = ntb["uid"]
        continuations = ntb.get("prefix_continuation")
        rm_scalar = batch.batch["rm_scores"].sum(dim=-1)

        bsz = len(uid)
        group_vals = defaultdict(list)
        for i in range(bsz):
            v = float(rm_scalar[i]) if action_type[i] != "malformed" else 0.0
            group_vals[uid[i]].append(v)
        parent_live = {u: (max(vs) != min(vs)) for u, vs in group_vals.items()}

        # evict entries whose replay diverged from the stored tool outcome
        failed_entries = ntb.get("prefix_rehydration_failed_entry")
        if failed_entries is not None:
            for eid in {e for e in failed_entries if e}:
                self.buffer.remove(eid)

        n_inserted = 0
        if continuations is not None:
            budget = self.buffer.max_inserts_per_step
            for i in range(bsz):
                cont = continuations[i]
                if cont is None or n_inserted >= budget:
                    continue
                if self.buffer.add(cont, global_steps, parent_live=parent_live.get(uid[i])):
                    n_inserted += 1

        behaviour_metrics(batch, metrics, self.buffer.max_depth)
        metrics.update(self.buffer.metrics())

    def before_advantage(self, batch: DataProto, metrics: dict):
        """Score the intrinsic-reward columns turn_ur and think_ur with the frozen probe."""
        from verl.trainer.ppo.turn_uncertainty import compute_prefix_turn_ur

        # on failure the estimator raises if beta_ur > 0 (no turn_ur column)
        try:
            turn_ur_col, think_ur_col, ur_metrics = compute_prefix_turn_ur(
                batch, self.tokenizer, self.config.algorithm.probe_model_path
            )
            batch.non_tensor_batch["turn_ur"] = turn_ur_col
            batch.non_tensor_batch["think_ur"] = think_ur_col
            metrics.update(ur_metrics)
        except Exception as err:
            print(f"[prefix_replay] turn_ur probe skipped this step: {err}")

    def after_advantage(self, batch: DataProto, metrics: dict):
        advantage_metrics(batch, self.config.algorithm, metrics)

    def save(self, global_steps: int):
        """Save the buffer next to the checkpoint of this step."""
        self.buffer.save_to_file(self._buffer_path(global_steps))


def _source_tag(data_source) -> str:
    ds = str(data_source or "")
    if "multiple_choice" in ds:
        return "mcq"
    if "charades" in ds:
        return "tg"
    return "other"


def _parse_span(obj):
    """Parse a [start, end] span from a GT field or tool-call arguments."""
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict):
            return float(obj["start_time"]), float(obj["end_time"])
        if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) == 2:
            return float(obj[0]), float(obj[1])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return None


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def behaviour_metrics(batch: DataProto, metrics: dict, max_depth: int):
    """``pr/*`` behaviour metrics: call rate per prefix depth and the IoU of the requested window on
    grounding tool rows."""
    ntb = batch.non_tensor_batch
    at = ntb["prefix_action_type"]
    n = len(at)
    depths = ntb["prefix_depth"]
    data_sources = ntb.get("data_source")
    uid = ntb.get("uid")
    continuations = ntb.get("prefix_continuation")
    reward_models = ntb.get("reward_model")

    def _d(i):
        return int(depths[i])

    def _src(i):
        return _source_tag(data_sources[i]) if data_sources is not None else "other"

    def _p_tool(rows):
        return (sum(1 for i in rows if at[i] == "tool") / len(rows)) if rows else None

    for d in range(max_depth):
        rows = [i for i in range(n) if _d(i) == d]
        if len(rows) >= 4:
            metrics[f"pr/policy/p_tool_d{d}"] = _p_tool(rows)
    for tag in ("mcq", "tg") if max_depth > 1 else ():
        rows = [i for i in range(n) if _d(i) == 1 and _src(i) == tag]
        if len(rows) >= 4:
            metrics[f"pr/policy/p_tool_d1_{tag}"] = _p_tool(rows)

    tool_rows = [i for i in range(n) if at[i] == "tool"]
    tiou_by_row: dict[int, float] = {}
    if continuations is not None and reward_models is not None:
        for i in tool_rows:
            if _src(i) != "tg" or continuations[i] is None:
                continue
            calls = continuations[i].get("tool_calls") or []
            gt = _parse_span((reward_models[i] or {}).get("ground_truth"))
            span = _parse_span(calls[-1].get("arguments")) if calls else None
            if gt is not None and span is not None:
                tiou_by_row[i] = _iou(span, gt)
    if tiou_by_row:
        metrics["pr/tool/tiou_mean"] = float(np.mean(list(tiou_by_row.values())))
        groups: dict[str, list[float]] = defaultdict(list)
        for i, v in tiou_by_row.items():
            if _d(i) == 0 and uid is not None:
                groups[uid[i]].append(v)
        gvals = [vs for vs in groups.values() if len(vs) >= 2]
        if gvals:
            metrics["pr/explore/group_tiou_spread"] = float(np.mean([max(vs) - min(vs) for vs in gvals]))


def advantage_metrics(batch: DataProto, algorithm_config, metrics: dict):
    """Metrics of a prefix-replay step: violation rates, raw reward per corpus and group ties, the trained
    per-row advantage (pr/adv/*) and the intrinsic reward's share of it."""
    ntb = batch.non_tensor_batch
    at = ntb["prefix_action_type"]
    n = len(at)
    rmask = batch.batch["response_mask"].float()
    denom = rmask.sum(dim=-1).clamp_min(1.0)
    row_adv = ((batch.batch["advantages"] * rmask).sum(dim=-1) / denom).cpu().numpy()
    scores = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
    raw = _raw_reward(algorithm_config, ntb["violation"], scores, metrics)

    ans = [i for i in range(n) if at[i] == "answer"]
    tool = [i for i in range(n) if at[i] == "tool"]
    if ans and tool:
        metrics["pr/adv/gap_tool_minus_answer"] = float(row_adv[tool].mean() - row_adv[ans].mean())
    metrics["pr/adv/abs_mean"] = float(np.abs(row_adv).mean())
    _corpus_metrics(ntb.get("extra_info"), raw, metrics)

    uid = ntb.get("uid")
    if uid is not None:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            groups[uid[i]].append(i)
        _tie_metrics(raw, groups, ntb.get("task_reward_func"), metrics)
        _ur_metrics(algorithm_config, raw, groups, at, ntb.get("turn_ur"), metrics)


def _raw_reward(algorithm_config, rules, scores, metrics: dict) -> np.ndarray:
    """The raw reward of ``compute_prefix_group_advantage`` from the recorded violations; logs one rate per
    rule."""
    n = len(rules)
    penalty = violations.violation_penalty(algorithm_config)
    raw = np.zeros(n, dtype=np.float64)
    n_violations = {}
    for i in range(n):
        if rules[i] is not None:
            raw[i] = -penalty
            n_violations[rules[i]] = n_violations.get(rules[i], 0) + 1
            continue
        raw[i] = float(scores[i])
    for rule in ("over_budget", "truncated", "tool_exec_error", "bad_format"):
        metrics[f"pr/violation/{rule}_rate"] = n_violations.get(rule, 0) / max(n, 1)
    metrics["pr/violation/any_rate"] = sum(n_violations.values()) / max(n, 1)
    return raw


def _corpus_metrics(extra_infos, raw: np.ndarray, metrics: dict):
    """Raw reward and row share per seed corpus (extra_info.corpus; "untagged" without one)."""
    if extra_infos is None:
        return
    n = len(raw)
    corpora = [str((extra_infos[i] or {}).get("corpus") or "untagged") for i in range(n)]
    for c in sorted(set(corpora)):
        rows_c = [i for i in range(n) if corpora[i] == c]
        metrics[f"pr/corpus/{c}/raw_reward_mean"] = float(raw[rows_c].mean())
        metrics[f"pr/corpus/{c}/row_share"] = len(rows_c) / n


def _tie_metrics(raw: np.ndarray, groups: dict, task_score, metrics: dict):
    """Share of groups (of >= 2 rows) whose raw reward ties exactly, split into all-wrong and all-right by
    the task score ``task_reward_func``."""
    scored = [g for g in groups.values() if len(g) >= 2]
    tied = [g for g in scored if float(raw[g].std()) < 1e-6]
    if scored:
        metrics["pr/group/frac_saturated"] = len(tied) / len(scored)
    if tied:
        means = [float(np.mean([float(task_score[i]) for i in g])) for g in tied]
        metrics["pr/group/frac_tied_wrong"] = sum(1 for m in means if m <= 1e-6) / len(scored)
        metrics["pr/group/frac_tied_right"] = sum(1 for m in means if m >= 1.0 - 1e-6) / len(scored)


def _ur_metrics(algorithm_config, raw: np.ndarray, groups: dict, at, turn_ur, metrics: dict):
    """ur_to_correctness_ratio = beta_ur * mean |z(ur)| / mean |z(r)| and ur_gap_tool_minus_answer = beta_ur *
    (mean z(ur) of tool rows - of answer rows) within a group. Approximate: population std, no winsorizing,
    violating rows included.
    """
    replay_cfg = algorithm_config.prefix_replay
    beta_ur = float(replay_cfg.beta_ur)
    norm_by_std = bool(algorithm_config.norm_adv_by_std_in_grpo)
    std_floor = float(replay_cfg.std_floor)
    ur_std_floor = float(replay_cfg.ur_std_floor)

    def _norm(vals: np.ndarray, floor: float) -> np.ndarray:
        c = vals - vals.mean()
        return c / max(float(vals.std()), floor) if norm_by_std else c

    n = len(raw)
    ur_col = None
    if turn_ur is not None:
        ur_col = np.array(
            [float(turn_ur[i]) if turn_ur[i] is not None else 0.0 for i in range(n)],
            dtype=np.float64,
        )
    corr_parts, ur_parts = [], []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        corr_parts.append(np.abs(_norm(raw[idxs], std_floor)))
        if ur_col is not None:
            m = [j for j, i in enumerate(idxs) if turn_ur[i] is not None]
            if len(m) >= 2:
                ur_parts.append(np.abs(_norm(ur_col[idxs][m], ur_std_floor)))
    if corr_parts:
        corr_abs = float(np.concatenate(corr_parts).mean())
        if ur_parts and beta_ur > 0.0 and corr_abs > 1e-8:
            ur_abs = float(np.concatenate(ur_parts).mean())
            metrics["pr/adv/ur_to_correctness_ratio"] = beta_ur * ur_abs / corr_abs

    ur_gaps = []
    if ur_col is not None and beta_ur > 0.0:
        for idxs in groups.values():
            mt = [j for j, i in enumerate(idxs) if at[i] == "tool" and turn_ur[i] is not None]
            ma = [j for j, i in enumerate(idxs) if at[i] == "answer" and turn_ur[i] is not None]
            if mt and ma:
                m = sorted(mt + ma)
                zu = _norm(ur_col[idxs][m], ur_std_floor)
                pos = {j: k for k, j in enumerate(m)}
                ur_gaps.append(float(zu[[pos[j] for j in mt]].mean() - zu[[pos[j] for j in ma]].mean()))
    if ur_gaps:
        metrics["pr/adv/ur_gap_tool_minus_answer"] = beta_ur * float(np.mean(ur_gaps))

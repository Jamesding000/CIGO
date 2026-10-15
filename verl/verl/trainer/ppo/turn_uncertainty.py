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
"""Intrinsic rewards from the uncertainty of a frozen probe model, and the annotation reward.

U at a response offset is the mean NLL of the content tokens of the gold answer y*, teacher-forced
after ``prompt + response[:offset]`` under the frozen model ``algorithm.probe_model_path``.
"""

from __future__ import annotations

import ast
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from cigo.reward import parse as ap

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


# y* phrasing is keyed by a substring of data_source
_MCQ_SUBSTRINGS = ("multiple_choice",)
_TEMPORAL_SUBSTRINGS = ("charades",)


def _format_y_star(data_source: str, ground_truth: Any) -> str:
    ds = str(data_source)
    if any(s in ds for s in _MCQ_SUBSTRINGS):
        return f"<answer>{str(ground_truth).strip()}</answer>"
    if any(s in ds for s in _TEMPORAL_SUBSTRINGS):
        if isinstance(ground_truth, (list, tuple)):
            start, end = float(ground_truth[0]), float(ground_truth[1])
        else:
            interval = ast.literal_eval(str(ground_truth))
            start, end = float(interval[0]), float(interval[1])
        # the phrasing the system prompt asks for
        return f"<answer>The event happens in {start:.2f} - {end:.2f} seconds.</answer>"
    raise ValueError(f"Unknown data_source for y* formatting: {data_source!r}")


# what the policy writes between </think> and <answer>; prepended to y* and not scored
_ANSWER_LEAD = "\n\n"


def _build_y_star(data_source: str, ground_truth: Any, tokenizer) -> tuple[list[int], list[int]]:
    """Return (y_star_ids, content_mask): the mask is 1 on tokens overlapping the text inside
    <answer>...</answer> (including a token merged across the tag boundary, e.g. ">The") and 0 on the
    tags and on ``_ANSWER_LEAD``.
    """
    text = _ANSWER_LEAD + _format_y_star(data_source, ground_truth)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    content_start = len(_ANSWER_LEAD) + len("<answer>")
    content_end = len(text) - len("</answer>")
    mask = [1 if (end > content_start and start < content_end and end > start) else 0 for start, end in offsets]
    return ids, mask


def _slice_video_mmi(mmi: dict, n_keep: int) -> dict:
    """multi_modal_inputs with only the first n_keep videos; images are kept in full."""
    sliced = dict(mmi)
    video_grid_thw = mmi.get("video_grid_thw")
    pixel_values_videos = mmi.get("pixel_values_videos")
    if video_grid_thw is None or pixel_values_videos is None:
        return sliced
    n_total = video_grid_thw.shape[0]
    if n_keep >= n_total:
        return sliced
    if n_keep <= 0:
        sliced.pop("pixel_values_videos", None)
        sliced.pop("video_grid_thw", None)
        return sliced
    # pixel_values_videos rows per video = T * H * W (unmerged patches).
    patches = (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]).long()
    offset = int(patches[:n_keep].sum().item())
    sliced["pixel_values_videos"] = pixel_values_videos[:offset]
    sliced["video_grid_thw"] = video_grid_thw[:n_keep]
    return sliced


@dataclass(frozen=True)
class ProbeSpec:
    """One teacher-forced uncertainty probe of row ``traj_idx``: it sees ``prompt + response[:anchor_offset]``.
    ``idx_within_kind`` is the occurrence of ``kind`` in the row (``after_think`` 0 is the first ``</think>``).
    """

    traj_idx: int
    kind: str
    idx_within_kind: int
    anchor_offset: int


@dataclass(frozen=True)
class Segment:
    """Reward of one step between two consecutive think boundaries (from -> to)."""

    from_probe: ProbeSpec
    to_probe: ProbeSpec
    r: float


def _find_think_close_positions(
    response_ids: torch.Tensor,
    assistant_spans: list[tuple[int, int]],
    think_close_token_ids: list[int],
) -> list[int]:
    """Response offsets just after each ``</think>`` inside an assistant turn, ascending.

    A probe at such an offset sees the state right after the think closes, before ``<tool_call>``
    or ``<answer>``. A turn without a closing tag (truncated) contributes none.
    """
    if not think_close_token_ids:
        return []
    seq = list(think_close_token_ids)
    L = len(seq)
    resp_list = response_ids.tolist() if isinstance(response_ids, torch.Tensor) else list(response_ids)
    positions: list[int] = []
    for span_start, span_end in assistant_spans:
        s, e = int(span_start), int(span_end)
        i = s
        while i + L <= e:
            if resp_list[i : i + L] == seq:
                positions.append(i + L)
                i += L
            else:
                i += 1
    positions.sort()
    return positions


def _enumerate_probes(
    bsz: int,
    real_responses: list[torch.Tensor],
    tool_spans_all,
    assistant_spans_all,
    tokenizer,
) -> list[list[ProbeSpec]]:
    """Per-row probes (prompt, every ``</think>`` close, every tool response end), sorted by offset."""
    think_seq = list(tokenizer.encode("</think>", add_special_tokens=False))
    probes_by_traj: list[list[ProbeSpec]] = []
    for i in range(bsz):
        specs = [ProbeSpec(traj_idx=i, kind="prompt", idx_within_kind=0, anchor_offset=0)]
        offsets = _find_think_close_positions(real_responses[i], list(assistant_spans_all[i]), think_seq)
        for j, off in enumerate(offsets):
            specs.append(ProbeSpec(traj_idx=i, kind="after_think", idx_within_kind=j, anchor_offset=off))
        for j, span in enumerate(tool_spans_all[i]):
            specs.append(ProbeSpec(traj_idx=i, kind="after_tool", idx_within_kind=j, anchor_offset=int(span[1])))
        specs.sort(key=lambda p: p.anchor_offset)
        probes_by_traj.append(specs)
    return probes_by_traj


_THINK_BOUNDARY_KINDS = ("prompt", "after_think")


def _build_reward_segments(probes: list[ProbeSpec], U: list[float]) -> list[Segment]:
    """Intrinsic reward per step: r_k = U(b_k) - U(b_{k+1}) for consecutive think boundaries b (the prompt,
    each ``</think>``). A tool block belongs to the step of the think that follows it; the rewards
    telescope to U(prompt) - U(last ``</think>``).
    """
    idxs = [k for k, p in enumerate(probes) if p.kind in _THINK_BOUNDARY_KINDS]
    return [
        Segment(from_probe=probes[a], to_probe=probes[b], r=U[a] - U[b]) for a, b in zip(idxs, idxs[1:], strict=False)
    ]


def _scatter_index(seg: Segment) -> int:
    """Response-token index credited with the segment's reward: the last token of the
    ``</think>`` that closes it."""
    return int(seg.to_probe.anchor_offset) - 1


_REFERENCE_MMI_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


def _span_iou(pred, gold) -> float:
    """Interval IoU as the grounding reward computes it; 0 when either side is None."""
    if pred is None or gold is None:
        return 0.0
    return max(0.0, ap.cal_iou_precision_recall(list(gold), list(pred))[0])


class _ReferenceProbeWorker:
    """Ray actor hosting the frozen probe model, kept on CPU between calls. One forward per probe."""

    def __init__(self, model_path: str, device_index: int = 0):
        import transformers

        n_dev = max(1, torch.cuda.device_count())
        self.device_index = device_index % n_dev
        self.device = f"cuda:{self.device_index}"

        config = transformers.AutoConfig.from_pretrained(model_path)
        cls = getattr(transformers, config.architectures[0])
        self.model = cls.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self.model.eval().requires_grad_(False)
        print(
            f"[probe-worker] shard {device_index} -> {self.device} (torch.cuda.device_count()={n_dev})",
            flush=True,
        )

    def score(
        self,
        prefixes: list[torch.Tensor],
        y_ids_per_probe: list[list[int]],
        probe_mmi_list: list[dict],
        max_y_len: int,
    ) -> torch.Tensor:
        """Return the log-probs of y*'s tokens, right-zero-padded to (N, max_y_len), on CPU."""
        N = len(prefixes)
        log_probs = torch.zeros((N, max_y_len), dtype=torch.float32)
        if N == 0:
            return log_probs
        torch.cuda.set_device(self.device_index)
        dev = self.device
        model = self.model.to(dev)
        with torch.no_grad():
            for idx in range(N):
                y = torch.tensor(y_ids_per_probe[idx], dtype=torch.long, device=dev)
                y_len = y.shape[0]
                input_ids = torch.cat([prefixes[idx].to(dev), y]).unsqueeze(0)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "logits_to_keep": y_len + 1,
                }
                for key, val in probe_mmi_list[idx].items():
                    val = val.to(dev)
                    inputs[key] = val.to(model.dtype) if val.is_floating_point() else val
                logits = model(**inputs).logits[0, :-1].float()
                logp = torch.log_softmax(logits, dim=-1)
                log_probs[idx, :y_len] = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).cpu()
        self.model.to("cpu")
        torch.cuda.empty_cache()
        return log_probs


_PROBE_WORKER_CACHE: dict[str, Any] = {}


def _get_probe_workers(model_path: str) -> list:
    """The probe actors (created once per process): PROBE_NUM_WORKERS (env, default 1), each on its own
    CUDA device. They reach the GPUs without claiming Ray GPU resources, so they must not spawn actors."""
    if model_path not in _PROBE_WORKER_CACHE:
        import ray

        n_workers = max(1, int(os.environ.get("PROBE_NUM_WORKERS", "1")))
        worker_cls = ray.remote(num_cpus=1)(_ReferenceProbeWorker)
        _PROBE_WORKER_CACHE[model_path] = [
            worker_cls.options(runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}).remote(
                model_path, i
            )
            for i in range(n_workers)
        ]
        logger.info("spawned %d frozen probe worker(s) for %s", n_workers, model_path)
    return _PROBE_WORKER_CACHE[model_path]


def _probe_model_path(path) -> str:
    """The frozen probe model; required, since the probes have no other scorer."""
    if not path:
        raise ValueError("probe_model_path is required: probes are scored by a frozen reference model")
    return str(path)


def _score_probes_reference(
    model_path: str,
    probe_specs: list[ProbeSpec],
    prefixes: list[torch.Tensor],
    probe_mmi_list: list[dict],
    y_star_ids: dict[int, list[int]],
    max_y_len: int,
) -> torch.Tensor:
    """Log-probs of y* under the frozen probe model, sharded over the probe workers."""
    import ray

    workers = _get_probe_workers(model_path)
    y_ids_per_probe = [y_star_ids[s.traj_idx] for s in probe_specs]
    mmi_slim = [{k: m[k] for k in _REFERENCE_MMI_KEYS if m.get(k) is not None} for m in probe_mmi_list]
    n_probes, n_workers = len(prefixes), len(workers)
    if n_workers == 1 or n_probes <= 1:
        return ray.get(workers[0].score.remote(prefixes, y_ids_per_probe, mmi_slim, max_y_len))

    # contiguous shards keep the probe order
    cuts = [round(n_probes * k / n_workers) for k in range(n_workers + 1)]
    futures = [
        workers[k].score.remote(
            prefixes[cuts[k] : cuts[k + 1]],
            y_ids_per_probe[cuts[k] : cuts[k + 1]],
            mmi_slim[cuts[k] : cuts[k + 1]],
            max_y_len,
        )
        for k in range(n_workers)
        if cuts[k + 1] > cuts[k]
    ]
    return torch.cat(ray.get(futures), dim=0)


def _real_tokens(batch: DataProto, rows) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """(prompt ids, response ids) of each listed row, without padding."""
    P = batch.batch["prompts"].shape[1]
    tokens = {}
    for i in rows:
        mask = batch.batch["attention_mask"][i]
        tokens[i] = (batch.batch["prompts"][i][mask[:P].bool()], batch.batch["responses"][i][mask[P:].bool()])
    return tokens


def _y_stars(batch: DataProto, rows, tokenizer) -> dict[int, tuple[list[int], list[int]]]:
    """(y* ids, content mask) of each listed row (``_build_y_star``)."""
    ntb = batch.non_tensor_batch
    return {i: _build_y_star(ntb["data_source"][i], ntb["reward_model"][i]["ground_truth"], tokenizer) for i in rows}


def _n_videos_at(sample_mmi: dict, counts, spans, anchor_offset: int) -> int:
    """Number of videos visible in a row's prefix at response offset ``anchor_offset``: all but those
    appended by tool turns ending after it (``counts``: videos per tool turn, parallel to ``spans``).
    """
    grid = sample_mmi.get("video_grid_thw") if sample_mmi else None
    total = int(grid.shape[0]) if grid is not None else 0
    if total == 0 or counts is None or not counts:
        return total
    n_done = sum(1 for span in spans if int(span[1]) <= anchor_offset)
    return total - int(sum(counts[n_done:]))


def _probe_u(probe_model_path, batch: DataProto, specs: list[ProbeSpec], tokens: dict, y_star: dict):
    """U = -(1/|y*_c|) * sum over content tokens t of log p(y*_t | prefix, y*_<t) for every probe, the
    prefix being the row's prompt and response up to the anchor, with the videos visible there.
    """
    ntb = batch.non_tensor_batch
    mmi_all = ntb.get("multi_modal_inputs", None)
    spans_all = ntb.get("tool_response_spans")
    counts_all = ntb.get("per_turn_video_counts", None)
    max_y_len = max(len(ids) for ids, _ in y_star.values())
    prefixes: list[torch.Tensor] = []
    probe_mmi_list: list[dict] = []
    content_mask = torch.zeros((len(specs), max_y_len), dtype=torch.float32)
    for idx, spec in enumerate(specs):
        i, off = spec.traj_idx, spec.anchor_offset
        prompt, response = tokens[i]
        prefixes.append(prompt if off == 0 else torch.cat([prompt, response[:off]]))
        sample_mmi = dict(mmi_all[i]) if mmi_all is not None and mmi_all[i] is not None else {}
        counts = None if counts_all is None else counts_all[i]
        spans = spans_all[i] if spans_all is not None else []
        probe_mmi_list.append(_slice_video_mmi(sample_mmi, _n_videos_at(sample_mmi, counts, spans, off)))
        cm = y_star[i][1]
        content_mask[idx, : len(cm)] = torch.tensor(cm, dtype=torch.float32)

    log_probs = _score_probes_reference(
        _probe_model_path(probe_model_path),
        specs,
        prefixes,
        probe_mmi_list,
        {i: ids for i, (ids, _) in y_star.items()},
        max_y_len,
    )
    cmask = content_mask.to(log_probs.device)
    return (((-log_probs) * cmask).sum(dim=1) / cmask.sum(dim=1).clamp_min(1.0)).detach().cpu()


def _col(bsz: int, values: dict) -> np.ndarray:
    """An object column of ``bsz`` rows: ``float(values[i])`` where given, else None."""
    out = np.empty(bsz, dtype=object)
    out[:] = None
    for i, v in values.items():
        out[i] = float(v)
    return out


def compute_turn_uncertainty_advantage(
    batch: DataProto,
    tokenizer,
    probe_model_path: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Intrinsic (process) reward advantage of the GRPO + IR arm, (A_proc[bsz, response_length], metrics):
    the step rewards r_k = U(b_k) - U(b_{k+1}) normalized within the group, summed over the steps closing at
    or after each token (``_segment_advantage``). A row without a closed ``</think>`` gets 0.
    """
    bsz = batch.batch["prompts"].shape[0]
    R = batch.batch["responses"].shape[1]
    ntb = batch.non_tensor_batch
    rows = range(bsz)
    tokens = _real_tokens(batch, rows)

    probes_by_traj = _enumerate_probes(
        bsz, [tokens[i][1] for i in rows], ntb["tool_response_spans"], ntb["assistant_turn_spans"], tokenizer
    )
    probe_specs: list[ProbeSpec] = [p for row in probes_by_traj for p in row]
    U = _probe_u(probe_model_path, batch, probe_specs, tokens, _y_stars(batch, rows, tokenizer))

    U_per: list[list[float]] = [[] for _ in range(bsz)]
    for idx, spec in enumerate(probe_specs):
        U_per[spec.traj_idx].append(float(U[idx]))

    reward_segs_by_traj = [_build_reward_segments(probes_by_traj[i], U_per[i]) for i in rows]
    A_proc = _segment_advantage(reward_segs_by_traj, ntb["uid"], R, batch.batch["prompts"].device)

    reward_seg_groups: dict[str, list[float]] = defaultdict(list)
    for segs in reward_segs_by_traj:
        for seg in segs:
            reward_seg_groups[f"{seg.from_probe.kind}__{seg.to_probe.kind}"].append(seg.r)
    metrics = {
        f"turn_uncertainty/reward_seg/{lbl}/r_nll": float(np.mean(vals)) for lbl, vals in reward_seg_groups.items()
    }

    return A_proc, metrics


def _normalize(raw: list[list[float]], uids, eps: float = 1e-6) -> list[list[float]]:
    """Normalized reward (r - mean_g) / (std_g + eps) over the pooled step rewards of each group g (rows
    sharing a uid); a group with fewer than 2 rewards or std <= eps gets zeros.
    """
    norm = [list(row) for row in raw]
    groups = defaultdict(list)
    for i in range(len(raw)):
        groups[str(uids[i])].append(i)
    for members in groups.values():
        pool = [v for i in members for v in raw[i]]
        if len(pool) < 2:
            for i in members:
                norm[i] = [0.0] * len(raw[i])
            continue
        mu, sd = float(np.mean(pool)), float(np.std(pool))
        for i in members:
            norm[i] = [0.0 if sd <= eps else (v - mu) / (sd + eps) for v in raw[i]]
    return norm


def _segment_advantage(segments_by_traj: list[list[Segment]], uids, R: int, device) -> torch.Tensor:
    """A_proc[i, t] = sum of the normalized step rewards (``_normalize``) of the steps of row i that close
    at or after token t (a step closes on the last token of its ``</think>``)."""
    norm = _normalize([[seg.r for seg in segs] for segs in segments_by_traj], uids)
    A_proc = torch.zeros((len(segments_by_traj), R), dtype=torch.float32)
    for i, segs in enumerate(segments_by_traj):
        for k, seg in enumerate(segs):
            tau = _scatter_index(seg)
            if 0 <= tau < R:
                A_proc[i, tau] += norm[i][k]
    return torch.flip(torch.cumsum(torch.flip(A_proc, dims=[-1]), dim=-1), dims=[-1]).to(device)


def answer_continuation_after_anchor(response_ids, think_close_id: int, think0_off: int):
    """Offset just after the last ``</think>`` of a ``[think0][call][result][think1]<answer>`` response;
    None when it does not lie past ``think0_off``.
    """
    pos = (response_ids == think_close_id).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None
    last = int(pos[-1].item()) + 1
    return last if last > think0_off else None


def _prefix_anchors(response, action_type, spans, think_close_id: int) -> list:
    """The (kind, offset) probe anchors of a prefix-replay row, or []: ``prompt`` at 0, ``after_think0``
    after the first ``</think>`` (before the first tool response on a tool row), and on a tool row
    ``after_think`` after the last ``</think>``.
    """
    anchors = [("prompt", 0)]
    pos = (response == think_close_id).nonzero(as_tuple=True)[0]
    if len(pos) > 0:
        off0 = int(pos[0].item()) + 1
        first_span_start = int(spans[0][0]) if len(spans) > 0 else None
        if action_type != "tool" or first_span_start is None or off0 <= first_span_start:
            anchors.append(("after_think0", off0))
    if action_type == "tool":
        think0_off = anchors[1][1] if len(anchors) > 1 else 0
        after_off = answer_continuation_after_anchor(response, think_close_id, think0_off)
        if after_off is not None:
            anchors.append(("after_think", after_off))
    return anchors if len(anchors) >= 2 else []


def compute_prefix_turn_ur(batch, tokenizer, probe_model_path: str):
    """Intrinsic reward of a prefix-replay row: turn_ur = -U(terminal), the terminal anchor being the last
    ``</think>`` before the row's answer (think0's on an answer row, think1's on a tool row). Equivalent
    to U(prompt) - U(terminal) after group centering. Returns (turn_ur, think_ur, metrics), the columns as
    object arrays; think_ur = U(prompt) - U(after think0) is logged only.
    """
    action_type = batch.non_tensor_batch.get("prefix_action_type")
    bsz = batch.batch["prompts"].shape[0]
    if action_type is None:
        return _col(bsz, {}), _col(bsz, {}), {}
    think_close_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]

    tool_idx = [i for i in range(bsz) if action_type[i] == "tool"]
    ans_idx = [i for i in range(bsz) if action_type[i] == "answer"]
    probe_rows = tool_idx + ans_idx
    if not probe_rows:
        return _col(bsz, {}), _col(bsz, {}), {}

    ntb = batch.non_tensor_batch
    tokens = _real_tokens(batch, probe_rows)
    y_star = _y_stars(batch, probe_rows, tokenizer)
    tool_spans_all = ntb.get("tool_response_spans")
    probe_specs: list[ProbeSpec] = []
    for i in probe_rows:
        spans = tool_spans_all[i] if tool_spans_all is not None else []
        for kind, off in _prefix_anchors(tokens[i][1], action_type[i], spans, think_close_id):
            probe_specs.append(ProbeSpec(traj_idx=i, kind=kind, idx_within_kind=0, anchor_offset=off))

    U = _probe_u(probe_model_path, batch, probe_specs, tokens, y_star)
    return _prefix_ur_columns(batch, probe_specs, U, action_type, tool_idx)


def _prefix_ur_columns(batch, probe_specs: list[ProbeSpec], U: torch.Tensor, action_type, tool_idx):
    """turn_ur, think_ur and pr/ur/frac_positive from the probes' U; writes the probe levels u_prompt,
    u_after_think0 and u_terminal onto ``batch``."""
    bsz = batch.batch["prompts"].shape[0]
    u_by_row: dict[int, dict[str, float]] = {}
    for idx, spec in enumerate(probe_specs):
        u_by_row.setdefault(spec.traj_idx, {})[spec.kind] = float(U[idx])
    think_ur: dict[int, float] = {}
    u_terminal: dict[int, float] = {}
    turn_ur_delta: dict[int, float] = {}  # tool rows, for the metrics only
    for i, u in u_by_row.items():
        if "prompt" in u and "after_think0" in u:
            think_ur[i] = u["prompt"] - u["after_think0"]

        if "after_think" in u:
            u_terminal[i] = u["after_think"]
            turn_ur_delta[i] = u.get("prompt", u.get("after_think0")) - u["after_think"]
        elif action_type[i] == "answer" and "after_think0" in u:
            u_terminal[i] = u["after_think0"]

    batch.non_tensor_batch["u_prompt"] = _col(bsz, {i: u["prompt"] for i, u in u_by_row.items() if "prompt" in u})
    batch.non_tensor_batch["u_after_think0"] = _col(
        bsz, {i: u["after_think0"] for i, u in u_by_row.items() if "after_think0" in u}
    )
    batch.non_tensor_batch["u_terminal"] = _col(bsz, u_terminal)

    metrics: dict[str, float] = {}
    dvals = [turn_ur_delta[i] for i in tool_idx if i in turn_ur_delta]
    if dvals:
        metrics["pr/ur/frac_positive"] = float(np.mean([v > 0 for v in dvals]))
    # -U(terminal): U(prompt) is constant within a group
    turn_ur = _col(bsz, {i: -u for i, u in u_terminal.items()})
    return turn_ur, _col(bsz, think_ur), metrics


def _annot_window(obj):
    """A ``(start, end)`` window from a ground-truth field or a tool-call argument; None otherwise (e.g. a
    multiple-choice letter)."""
    try:
        if isinstance(obj, str):
            obj = ast.literal_eval(obj)
        if isinstance(obj, dict):
            return float(obj["start_time"]), float(obj["end_time"])
        if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) == 2:
            return float(obj[0]), float(obj[1])
    except (ValueError, TypeError, KeyError, SyntaxError):
        return None
    return None


def compute_annotation_segment_advantage(batch: DataProto, tokenizer):
    """Annotation reward advantage of the GRPO + AR arm, (A_proc[bsz, response_length], metrics): as
    ``compute_turn_uncertainty_advantage`` with step reward r_k = max IoU(requested clip, annotated window)
    over the step's calls, or IoU([0, duration], window) for a step without a call (after Video-MTR,
    arXiv 2508.20478). Rows without an annotated window get 0.
    """
    bsz = batch.batch["prompts"].shape[0]
    R = batch.batch["responses"].shape[1]

    assistant_spans_all = batch.non_tensor_batch["assistant_turn_spans"]
    tool_spans_all = batch.non_tensor_batch["tool_response_spans"]
    reward_model_all = batch.non_tensor_batch["reward_model"]
    uid_all = batch.non_tensor_batch["uid"]
    windows_all = batch.non_tensor_batch.get("tool_call_windows")
    if windows_all is None:
        raise ValueError(
            "algorithm.annotation_reward.enable=True but non_tensor_batch has no "
            "'tool_call_windows' column — ToolAgentLoop is not exporting the requested "
            "clip windows, so no turn can be scored against the annotation."
        )
    durations_all = batch.non_tensor_batch.get("video_duration")
    if durations_all is None:
        raise ValueError(
            "algorithm.annotation_reward.enable=True but non_tensor_batch has no "
            "'video_duration' column — without it the no-call baseline collapses to 0 and "
            "the reward becomes a per-call bonus. ToolAgentLoop must export it."
        )

    tokens = _real_tokens(batch, range(bsz))
    real_responses = [tokens[i][1] for i in range(bsz)]
    probes_by_traj = _enumerate_probes(bsz, real_responses, tool_spans_all, assistant_spans_all, tokenizer)

    segments_by_traj: list[list[Segment]] = []
    n_annotated = 0
    n_no_duration = 0
    call_beats_base: list[bool] = []
    for i in range(bsz):
        gt = _annot_window((reward_model_all[i] or {}).get("ground_truth"))
        probes = probes_by_traj[i]
        bounds = [k for k, p in enumerate(probes) if p.kind in _THINK_BOUNDARY_KINDS]
        if gt is None or len(bounds) < 2:
            segments_by_traj.append([])
            continue
        try:
            duration = float(durations_all[i])
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0.0:
            n_no_duration += 1
            segments_by_traj.append([])
            continue
        # an ndarray: `or []` would raise
        w = windows_all[i]
        windows = list(w) if w is not None else []
        # skip a row whose windows and tool spans are misaligned
        if len(windows) != len(tool_spans_all[i]):
            segments_by_traj.append([])
            continue
        n_annotated += 1
        base = _span_iou((0.0, duration), gt)
        segs: list[Segment] = []
        for a, b in zip(bounds, bounds[1:], strict=False):
            # the step's calls: after_tool probes between its two boundaries (indexes tool_call_windows)
            inside = [probes[k].idx_within_kind for k in range(a + 1, b + 1) if probes[k].kind == "after_tool"]
            ious = [_span_iou(_annot_window(windows[c]), gt) for c in inside if c < len(windows)]
            call_beats_base.extend(v > base for v in ious)
            r = max(ious) if ious else base
            segs.append(Segment(from_probe=probes[a], to_probe=probes[b], r=r))
        segments_by_traj.append(segs)

    if n_no_duration > 0 and n_annotated == 0:
        raise ValueError(
            f"annotation reward: {n_no_duration} rows carry a ground-truth window but NONE "
            "has a usable video duration, so no no-call baseline can be computed. The "
            "'video_duration' column is present but empty — check ToolAgentLoop's export."
        )

    A_proc = _segment_advantage(segments_by_traj, uid_all, R, batch.batch["prompts"].device)

    metrics = {
        "annotation_reward/frac_rows_annotated": n_annotated / max(1, bsz),
        "annotation_reward/frac_calls_beating_baseline": (float(np.mean(call_beats_base)) if call_beats_base else 0.0),
    }
    return A_proc, metrics

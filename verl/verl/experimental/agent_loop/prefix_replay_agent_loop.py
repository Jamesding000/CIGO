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
"""Prefix-replay agent loop for single-step RL with a replay buffer.

A row optionally starts from a stored prefix (``prefix_state``: its pre-splice ids become the prompt and its tool
videos are rehydrated by re-running its calls), generates one decision turn (a tool call is rolled one more step
to its answer), and emits a ``prefix_continuation`` snapshot of its first new tool turn for the buffer. The replayed
prefix is on the prompt side of the loss boundary, so stale tokens are never trained (arXiv 2605.05112).
"""

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def expanded_length(agent_data: AgentData) -> int:
    """Length of `prompt_ids` once every video placeholder is spliced into its expansion."""
    return len(agent_data.prompt_ids) + sum(max(0, len(e) - 3) for e in agent_data.turn_state.per_video_expansion if e)


@register("prefix_replay_agent")
class PrefixReplayAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ceiling on tool calls per question; a row at depth k is budgeted min(k + 1, ceiling)
        self.tool_budget_max = int(self.config.algorithm.prefix_replay.max_depth)
        self._init_remaining_tails()

    def _init_remaining_tails(self) -> None:
        """Token tables for _rewrite_remaining_counts.

        A tool result ends with REMAINING_MARKER and a remaining-calls sentence (cigo.prompts.remaining_tail).
        A replayed row overwrites each sentence in place, so all sentences must tokenize to one length. The
        marker tokenizes differently after a newline, a space or nothing, so each encoding is matched.
        """
        from cigo import prompts

        variants = []
        for pre in ("\n", " ", ""):
            v = self.tokenizer.encode(pre + prompts.REMAINING_MARKER, add_special_tokens=False)
            if v and v not in variants:
                variants.append(v)
        self._remaining_marker_ids = variants
        tails = {
            k: self.tokenizer.encode(prompts.remaining_tail(k), add_special_tokens=False)
            for k in range(self.tool_budget_max + 1)
        }
        tlens = {len(v) for v in tails.values()}
        assert len(tlens) == 1, (
            "cigo.prompts.remaining_tail(k) must tokenize to a COMMON length for every k "
            "so the replayed tool result can be corrected without shifting any offset; got "
            f"{sorted((k, len(v)) for k, v in tails.items())}"
        )
        self._remaining_tail_ids = tails
        self._remaining_tail_len = tlens.pop()
        # A real tail is followed by the close of its tool result; a quote of one in the policy's text is not.
        self._tool_response_close_id = self.tokenizer.convert_tokens_to_ids("</tool_response>")
        assert self._tool_response_close_id not in (None, self.tokenizer.unk_token_id), (
            "</tool_response> must be a single token for _rewrite_remaining_counts"
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        self._prefix_state = kwargs.get("prefix_state", None)
        self._row_kwargs = kwargs
        self._first_turn_snapshotted = False
        self._action_type = None
        # Set once the tool result is spliced: the next generation is the tool branch's answer step
        self._answer_cont_pending = False
        self._answer_cont_done = False
        self._prefix_depth = 0
        self._rehydration_failed_entry = None
        self._over_budget = False  # model emitted a call with no budget left
        self._remaining_fixed = 0  # stale remaining-calls tails corrected on this row
        self._budget_declared = None  # what the prompt told this row
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["prefix_action_type"] = self._action_type or "malformed"
        output.extra_fields["prefix_depth"] = self._prefix_depth
        # The buffer entry this row was replayed from (None on fresh roots)
        output.extra_fields["prefix_entry_id"] = (self._prefix_state or {}).get("entry_id")
        output.extra_fields.setdefault("tool_exec_error", False)
        output.extra_fields["prefix_rehydration_failed_entry"] = self._rehydration_failed_entry
        # Every per-row field needs a default: worker chunks that disagree on columns break DataProto.concat
        output.extra_fields.setdefault("prefix_continuation", None)
        output.extra_fields["over_budget"] = self._over_budget
        output.extra_fields["remaining_fixed"] = self._remaining_fixed
        output.extra_fields["budget_declared"] = self._budget_declared
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        ps = self._prefix_state
        if ps is None:
            # Fresh root (depth 0): one call, then answer
            state = await super()._handle_pending_state(agent_data, sampling_params)
            self._declare_tool_budget(agent_data)
            return state
        n_init_vids = len(agent_data.video_data or [])
        try:
            return await self._seed_from_prefix(agent_data, ps)
        except Exception as e:
            # A failed rehydration degrades this row to a fresh depth-0 row; the trainer evicts the entry
            logger.warning(
                f"prefix replay rehydration failed for entry {ps.get('entry_id')}: {e}; "
                "degrading to a fresh depth-0 row"
            )
            self._rehydration_failed_entry = ps.get("entry_id")
            self._prefix_state = None
            agent_data.video_data = agent_data.video_data[:n_init_vids] if n_init_vids else None
            agent_data.turn_state.per_video_expansion = []
            agent_data.prompt_ids = []
            agent_data.user_turns = 0
            agent_data.assistant_turns = 0
            self._prefix_depth = 0
            state = await super()._handle_pending_state(agent_data, sampling_params)
            self._declare_tool_budget(agent_data)
            return state

    def _declare_tool_budget(self, agent_data: AgentData) -> None:
        """Set this row's budget, min(depth + 1, tool_budget_max), and make the replayed tool results state the
        remaining-calls counts that follow from it."""
        budget = min(self._prefix_depth + 1, self.tool_budget_max)
        self._budget_declared = budget
        self._remaining_fixed = self._rewrite_remaining_counts(agent_data, budget)

    def _rewrite_remaining_counts(self, agent_data: AgentData, budget: int) -> int:
        """Correct every stale remaining-calls sentence the prefix carries, in place.

        After the i-th tool result the row has ``budget - (i+1)`` calls left. Returns how many sentences were
        corrected.
        """
        ids = agent_data.prompt_ids
        n, tlen = len(ids), self._remaining_tail_len
        # Where each tail starts, under any marker encoding (two encodings can match one occurrence)
        starts = set()
        for marker in self._remaining_marker_ids:
            mlen = len(marker)
            for j in range(n - mlen):
                if ids[j : j + mlen] == marker:
                    starts.add(j + mlen)
        # real tails only: a quoted marker would shift every later count by one
        starts = {p for p in starts if self._tool_response_close_id in ids[p + tlen : p + tlen + 2]}
        fixed = 0
        for i, pos in enumerate(sorted(starts)):
            if pos + tlen > n:
                continue
            left = max(0, min(budget - (i + 1), self.tool_budget_max))
            ids[pos : pos + tlen] = list(self._remaining_tail_ids[left])
            fixed += 1
        return fixed

    def _score_partial_format(self, agent_data: AgentData):
        """Valid-so-far format score of the decision turn, for both branches; the first call wins.

        Must run before the tool result is spliced (result text between tags fails the no-gap check). Special
        tokens are skipped (<|im_end|> would read as a foreign tag).
        """
        if "partial_format" in agent_data.extra_fields:
            return
        from cigo.reward.format_gate import format_partial_reward

        region = agent_data.prompt_ids[len(agent_data.prompt_ids) - len(agent_data.response_mask) :]
        text = self.tokenizer.decode(region, skip_special_tokens=True)
        agent_data.extra_fields["partial_format"] = float(format_partial_reward(text))

    async def _seed_from_prefix(self, agent_data: AgentData, ps: dict) -> AgentState:
        # Stored ids are pre-splice and end with the assistant header and the auto-opened <think>
        agent_data.prompt_ids = [int(t) for t in ps["prefix_ids"]]
        self._cache_video_expansions(agent_data, agent_data.video_data)

        # Rehydrate the tool videos by re-executing the stored calls in order
        for call in ps.get("tool_calls", []):
            tool_response, _, _ = await self._call_tool(
                FunctionCall(name=call["name"], arguments=call["arguments"]),
                agent_data.tools_kwargs,
                agent_data,
            )
            videos = tool_response.video or []
            if videos:
                agent_data.video_data = (agent_data.video_data or []) + videos
            self._cache_video_expansions(agent_data, videos)

        video_pad_id = getattr(self.processor, "video_token_id", None)
        n_pads = sum(1 for t in agent_data.prompt_ids if t == video_pad_id)
        n_videos = len(agent_data.video_data or [])
        if n_pads != n_videos:
            raise RuntimeError(
                f"rehydration mismatch: {n_pads} video placeholders in stored "
                f"prefix but {n_videos} rehydrated videos (entry tool_calls={ps.get('tool_calls')})"
            )
        # tokenizer.pad never truncates; buffer admission keeps the expanded prefix within prompt_length
        expanded_len = expanded_length(agent_data)
        if expanded_len > self.prompt_length:
            raise RuntimeError(
                f"expanded prefix length {expanded_len} exceeds prompt_length "
                f"{self.prompt_length}; buffer admission_margin is too small"
            )

        self._prefix_depth = int(ps.get("depth", 0))
        # A prefix at the ceiling has no call left and must answer this turn
        self._declare_tool_budget(agent_data)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        next_state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if self._answer_cont_pending:
            return self._finish_answer_continuation(agent_data)
        if next_state == AgentState.TERMINATED:
            # The decision turn is final (answer, malformed or cap-preempted); tool rows are scored pre-splice
            self._score_partial_format(agent_data)
        if self._action_type is None and next_state == AgentState.TERMINATED:
            await self._classify_final_turn(agent_data)
        return next_state

    def _finish_answer_continuation(self, agent_data: AgentData) -> AgentState:
        """The tool branch's answer step just ran. The row's action stays "tool"; the answer supplies its reward."""
        self._answer_cont_pending = False
        self._answer_cont_done = True
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        # With no stop strings, an unclosed answer means the turn hit its cap
        answer_closed = "</answer>" in text
        # The row's one call is spent: a call here instead of an answer is over budget
        if not answer_closed:
            spans = agent_data.turn_state.assistant_turn_spans
            turn_start = spans[-1][0] if spans else 0
            tail = self.tokenizer.decode(agent_data.response_ids[turn_start:], skip_special_tokens=True)
            if "<tool_call>" in tail:
                self._over_budget = True
        agent_data.tool_calls = []  # a one-step unit takes no second tool step
        self._action_type = "tool"
        return AgentState.TERMINATED

    async def _classify_final_turn(self, agent_data: AgentData) -> None:
        """The first turn ended without executing a tool: an answer, a cap-preempted call, or malformed."""
        text = self.tokenizer.decode(agent_data.response_ids)
        if "<answer>" in text:
            self._action_type = "answer"
        else:
            # The turn cap fires before the tool parse, so a call at the cap is classified by content
            calls = await self._extract_tool_calls(agent_data, agent_data.response_ids)
            self._action_type = "tool" if calls else "malformed"

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        executed_calls = [
            {"name": c.name, "arguments": c.arguments} for c in agent_data.tool_calls[: self.max_parallel_calls]
        ]
        # The tool-response envelope (call index, windows so far, calls remaining); replayed calls count as history
        from cigo import prompts

        parent_calls = list((self._prefix_state or {}).get("tool_calls", []))
        duration = agent_data.tools_kwargs.get("duration")
        agent_data.tools_kwargs.update(
            prompts.envelope_kwargs(
                parent_calls,
                budget=self._budget_declared,
                duration=float(duration) if duration is not None else None,
            )
        )

        self._score_partial_format(agent_data)
        next_state = await super()._handle_processing_tools_state(agent_data)

        if self._action_type is None:
            self._action_type = "tool"

        s, e = agent_data.turn_state.tool_response_spans[-1]
        region = agent_data.prompt_ids[len(agent_data.prompt_ids) - len(agent_data.response_mask) :]
        tool_text = self.tokenizer.decode(region[s:e])
        if "Error when executing tool" in tool_text or "tool execution failed" in tool_text:
            agent_data.extra_fields["tool_exec_error"] = True

        if not self._first_turn_snapshotted:
            self._first_turn_snapshotted = True
            # The buffer stores <prefix><think><tool_call><result>, without the answer step
            self._snapshot_continuation(agent_data, executed_calls)
            # One more step to the answer, nothing injected, so both actions are scored by the same reward
            self._answer_cont_pending = True
            return AgentState.GENERATING
        return next_state

    def _snapshot_continuation(self, agent_data: AgentData, executed_calls: list[dict]):
        ps = self._prefix_state or {}
        parent_calls = list(ps.get("tool_calls", []))
        prefix_ids = list(agent_data.prompt_ids)
        row = {}
        for key in ("raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "index"):
            if key in self._row_kwargs:
                row[key] = self._row_kwargs[key]
        agent_data.extra_fields["prefix_continuation"] = {
            "row": row,
            "prefix_ids": prefix_ids,
            "tool_calls": parent_calls + executed_calls,
            "user_turns": int(ps.get("user_turns", 0)) + 1,
            "assistant_turns": int(ps.get("assistant_turns", 0)) + 1,
            "depth": int(ps.get("depth", 0)) + 1,
            "expanded_len": expanded_length(agent_data),
            # The buffer admits a prefix only if its decision turn was well formed and within budget
            "partial_format": agent_data.extra_fields.get("partial_format"),
            "over_budget": bool(self._over_budget),
        }

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        raise NotImplementedError("prefix_replay_agent does not support interactions")

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
"""Video tool rollout hooks for verl's ToolAgentLoop.

VideoToolMixin holds per-turn token caps, the turn close spliced before a tool result, video tool results with
their Qwen3-VL expansions, and the per-turn spans and counters the trainer reads from extra_fields. Per-row
state lives in `agent_data.turn_state`.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.experimental.agent_loop import qwen3vl_video
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput


@dataclass
class VideoTurnState:
    """Per-row rollout state. Spans are in pre-splice response-token coordinates; the postprocess splice
    remaps them."""

    n_truncated_turns: int = 0  # assistant turns that hit their token cap
    over_budget: bool = False  # its last turn, at the assistant-turn cap, called a tool instead of answering
    per_video_expansion: list[list[int]] = field(default_factory=list)  # parallel to video_data
    assistant_turn_spans: list[tuple[int, int]] = field(default_factory=list)
    tool_response_spans: list[tuple[int, int]] = field(default_factory=list)
    # The videos each tool turn added and the window its call asked for, both parallel to tool_response_spans
    per_turn_video_counts: list[int] = field(default_factory=list)
    tool_call_windows: list[Optional[tuple[float, float]]] = field(default_factory=list)


def clip_spans(spans: list[tuple[int, int]], cap: int) -> list[tuple[int, int]]:
    """The spans that start before `cap`, their ends clipped to it."""
    clipped: list[tuple[int, int]] = []
    for s, e in spans:
        if s >= cap:
            break
        clipped.append((s, min(e, cap)))
    return clipped


def first_call_window(calls) -> Optional[tuple[float, float]]:
    """`(start_time, end_time)` of the first parseable clip call, else None (parsed as in evaluation)."""
    from cigo import prompts

    wins = prompts.call_windows([{"arguments": c.arguments} for c in calls or []])
    return wins[0] if wins else None


class VideoToolMixin:
    """Hooks ToolAgentLoop calls at fixed points of its state machine, and helpers both tool loops use."""

    def _init_video_tool(self) -> None:
        # Per-turn cap on generated tokens
        self.single_turn_response_length = self.rollout_config.single_turn_response_length
        # The turn close the chat template writes after an assistant message; none without a ChatML <|im_end|>
        has_im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>") not in (None, self.tokenizer.unk_token_id)
        self._turn_close_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False) if has_im_end else []

    async def apply_chat_template(
        self, messages, tools=None, images=None, videos=None, remove_system_prompt: bool = False
    ) -> list[int]:
        """Qwen3-VL: the prompt keeps one raw placeholder per video, which vLLM's prompt replacement
        expands per frame itself. Other processors go through the base class."""
        if not qwen3vl_video.is_qwen3vl(self.processor):
            return await super().apply_chat_template(messages, tools, images, videos, remove_system_prompt)
        assert not images, "image inputs are not supported on the Qwen3-VL video path (only videos are expanded)"
        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.processor,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        prompt_ids = normalize_token_ids(
            self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        )
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        return prompt_ids

    def _cache_video_expansions(self, agent_data, videos: list) -> None:
        """Append the expansion of each of `videos`, in the order they entered `video_data`."""
        agent_data.turn_state.per_video_expansion += [
            qwen3vl_video.video_expansion_ids(self.processor, v, m) for v, m in videos or []
        ]

    def _append_harness_ids(self, agent_data, ids: list[int], mask: int) -> None:
        """Append tokens the harness wrote, not the policy, to the context and the response region: `mask` 1
        trains them, 0 masks them. Recorded log-probs get 0.0 for them."""
        agent_data.prompt_ids += ids
        agent_data.response_mask += [mask] * len(ids)
        if self.rollout_config.calculate_log_probs or agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(ids)

    async def _extract_tool_calls(self, agent_data, ids: list[int]) -> list:
        _, calls = await self.tool_parser.extract_tool_calls(ids, [tool.tool_schema for tool in self.tools.values()])
        return calls

    def _turn_sampling_params(self, agent_data, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Per-turn token cap; the total response is checked at postprocess."""
        return {**sampling_params, "max_tokens": self.single_turn_response_length}

    async def _cap_turn(self, agent_data, output: TokenOutput, sampling_params: dict[str, Any]) -> TokenOutput:
        """Count a turn that reached its cap."""
        if len(output.token_ids) >= sampling_params["max_tokens"]:
            agent_data.turn_state.n_truncated_turns += 1
        return output

    def _record_assistant_turn(self, agent_data) -> None:
        """The span of the turn just appended to the response."""
        end = len(agent_data.response_mask)
        agent_data.turn_state.assistant_turn_spans.append((end - len(agent_data.response_ids), end))

    def _on_assistant_turn_cap(self, agent_data) -> None:
        """The rollout stops at the assistant-turn cap: a last turn that calls again instead of answering is
        over budget (scored, not prevented)."""
        text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        if "</answer>" not in text and "<tool_call>" in text:
            agent_data.turn_state.over_budget = True

    def _close_assistant_turn(self, agent_data) -> None:
        """Splice the `<|im_end|>\\n` the chat template would have written, masked like the tool response.
        No-op when the turn already ends with it."""
        if agent_data.prompt_ids[-len(self._turn_close_ids) :] != self._turn_close_ids:
            self._append_harness_ids(agent_data, self._turn_close_ids, 0)

    def _after_tool_turn(self, agent_data, videos: list, n_ids: int) -> None:
        """Record the tool turn of `n_ids` tokens just appended: its videos (with their expansions; the ids hold
        one raw placeholder each), its span and video count, and the window of the call it executed."""
        st = agent_data.turn_state
        if videos:  # a row without videos keeps video_data None
            agent_data.video_data = (agent_data.video_data or []) + videos
        self._cache_video_expansions(agent_data, videos)
        end = len(agent_data.response_mask)
        st.tool_response_spans.append((end - n_ids, end))
        st.per_turn_video_counts.append(len(videos))
        st.tool_call_windows.append(first_call_window(agent_data.tool_calls[: self.max_parallel_calls]))

    def _video_outputs(self, agent_data) -> dict[str, Any]:
        """Our extra_fields: tool and cap counters, the expansions for the postprocess splice, and the spans
        clipped to response_length with their parallel lists."""
        st = agent_data.turn_state
        out = {
            "valid_tool_call_stats": len(agent_data.tool_calls),
            "valid_tool_exec_stats": sum(1 for r in agent_data.tool_rewards if r is not None),
            "n_truncated_turns": st.n_truncated_turns,
            "over_budget": st.over_budget,  # PrefixReplayAgentLoop.run overwrites it
        }
        if st.per_video_expansion:
            out["per_video_expansion"] = st.per_video_expansion
        out["assistant_turn_spans"] = clip_spans(st.assistant_turn_spans, self.response_length)
        tool_spans = clip_spans(st.tool_response_spans, self.response_length)
        out["tool_response_spans"] = tool_spans
        out["per_turn_video_counts"] = st.per_turn_video_counts[: len(tool_spans)]
        out["tool_call_windows"] = st.tool_call_windows[: len(tool_spans)]
        # tools_kwargs does not reach the trainer; the annotation reward's no-call baseline needs the duration
        out["video_duration"] = float(agent_data.tools_kwargs.get("duration") or 0.0)
        return out

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""The format gate of both scorers.

`pure_format_score` is the strict verdict a malformed row is penalized on: the tag structure
(`format_toolrwd_reward`) and the content checks (`outside_text_ok`, `tool_calls_parseable`, `answer_payload_ok`).
`format_partial_reward` is the mid-trace check of the prefix-replay buffer.
"""

import json
import re

from . import parse as ap

# Tags a well-formed trace may end on.
TERMINAL_TAGS = ("answer",)

_TAG = re.compile(r"</?(?:think|tool_call|answer)>")
_BLOCK = re.compile(r"<(.*?)>(.*?)</\1>", re.DOTALL)


def format_toolrwd_reward(content):
    """1 when the response, normalised by `normalize_multiturn_response`, is a run of think / tool_call /
    answer blocks that opens with think, never repeats a tag back to back, has only whitespace between
    blocks and ends on a terminal tag; else 0."""
    content = normalize_multiturn_response(content)
    tag_list = []
    last_end = None
    for match in _BLOCK.finditer(content):
        tag = match.group(1)
        if tag not in ("think", "tool_call", "answer"):
            return 0
        if tag_list and tag_list[-1] == tag:
            return 0
        if last_end is not None and content[last_end: match.start()].strip():
            return 0
        tag_list.append(tag)
        last_end = match.end()
    return 1 if tag_list and tag_list[0] == "think" and tag_list[-1] in TERMINAL_TAGS else 0


def outside_text_ok(predict_str: str) -> bool:
    """True when the response opens with a tag, or with text whose first tag is `</think>` (a response that
    starts inside the think the prompt opened). Trailing text is not checked."""
    tags = list(_TAG.finditer(predict_str or ""))
    if not tags:
        return False
    if not predict_str[: tags[0].start()].strip():
        return True
    return tags[0].group(0) == "</think>"


def answer_payload_ok(predict_str: str, task: str, extra_info: dict) -> bool:
    """True when the answer span carries a parsable answer: an option letter (task="mcq"; a non-empty span when
    the question prints no options) or a [start, end] pair (task="tg"). An eval-mode read counts when it comes
    from inside the answer tag.
    """
    def _ok(reward_parse, eval_parse):
        if reward_parse.value:
            return True
        return bool(eval_parse.value) and bool(eval_parse.in_answer_tag)

    if task == "tg":
        return _ok(ap.tg_for_reward(predict_str), ap.tg_for_eval(predict_str))
    question = extra_info["question"]
    if ap.parse_options(question):
        return _ok(ap.mcq_for_reward(predict_str, question), ap.mcq_for_eval(predict_str, question))
    return bool(ap.free_text_span(predict_str).strip())


def pure_format_score(predict_str: str, task: str, extra_info: dict) -> float:
    """1.0 when the tag structure and every content check pass, else 0.0."""
    ok = (format_toolrwd_reward(predict_str) > 0 and outside_text_ok(predict_str)
          and tool_calls_parseable(predict_str) and answer_payload_ok(predict_str, task, extra_info))
    return 1.0 if ok else 0.0


def normalize_multiturn_response(content: str) -> str:
    """Prepend the `<think>` the prompt opened (when missing) and drop the injected tool-response turns."""
    content = content or ""
    if not content.lstrip().startswith("<think>"):
        content = "<think>" + content
    return re.sub(r"</tool_call>.*?<think>", "</tool_call>\n<think>", content, flags=re.DOTALL)


def format_partial_reward(content):
    """1 when a prefix-replay decision turn is well formed so far, else 0.

    Unlike ``format_toolrwd_reward`` the turn may end at ``</tool_call>`` and may start inside the prompt's
    think. It needs at least one complete, non-empty think / tool_call / answer block, no nested or adjacent
    duplicate tags, no text outside the blocks, and nothing after an answer.
    """
    if "</think>" in content and "<think>" not in content.split("</think>", 1)[0]:
        content = "<think>" + content
    tag_list = []
    first_start = None
    last_pos = None
    for match in _BLOCK.finditer(content):
        tag = match.group(1)
        body = match.group(2)
        if tag not in ("think", "tool_call", "answer"):
            return 0
        # the non-greedy pair regex would read "<think>r <answer>B</answer> more</think>" as one think block
        if _TAG.search(body):
            return 0
        if not body.strip():
            return 0
        if tag_list and tag_list[-1] == tag:
            return 0
        if last_pos and content[last_pos[1]: match.start()].strip():
            return 0
        tag_list.append(tag)
        if first_start is None:
            first_start = match.start()
        last_pos = (match.start(), match.end())
    if not tag_list:
        return 0
    if "answer" in tag_list and tag_list[-1] != "answer":
        return 0
    if content[:first_start].strip():
        return 0
    if content[last_pos[1]:].strip():
        return 0
    return 1


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def tool_calls_parseable(predict_str: str) -> bool:
    """True when every <tool_call> body is a JSON object with a non-empty name and arguments (an object, or a
    list of objects) whose start_time / end_time convert to float; a turn with no call passes.
    """
    for body in _TOOL_CALL_RE.findall(predict_str or ""):
        body = body.strip()
        if not body:
            return False
        try:
            payload = json.loads(body)
        except Exception:
            return False
        if not isinstance(payload, dict) or not str(payload.get("name") or "").strip():
            return False
        args = payload.get("arguments")
        items = args if isinstance(args, list) else [args]
        if not items:
            return False
        for a in items:
            if not isinstance(a, dict):
                return False
            try:
                float(a["start_time"])
                float(a["end_time"])
            except (KeyError, TypeError, ValueError):
                return False
    return True

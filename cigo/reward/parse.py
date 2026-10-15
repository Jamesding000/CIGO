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
"""Reading a model's answer out of its response: multiple choice and temporal grounding.

The parser reports what the model answered and on what evidence (``rule``, ``scope``, ``in_answer_tag``).
``Mode`` controls one thing: the reward refuses a guess (an unanchored rule, or a trace that never terminated),
evaluation admits it. The rules follow lmms-eval and VLMEvalKit, except that a span merely opening with an option
letter is a guess and grounding takes the last timestamp match, not the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

# From lmms-eval (lmms_eval/tasks/_task_utils/mcq_extract.py), verbatim.
_DEFAULT_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H"]

_ANSWER_PHRASES = [
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
    # Korean
    "옵션",
    "정답은",
    "답은",
    "답:",
    # Chinese
    "答案是",
    "答案为",
    "选",
    # Japanese
    "答えは",
]

_EXTRA_PHRASES = ("answer:",)
_PHRASES = tuple(_ANSWER_PHRASES) + _EXTRA_PHRASES

__all__ = [
    "Mode",
    "Parsed",
    "select_span",
    "parse_options",
    "parse_mcq",
    "parse_tg",
    "gold_letter",
    "gold_span",
    "cal_iou_precision_recall",
    "free_text_span",
    "mcq_for_reward",
    "mcq_for_eval",
    "tg_for_reward",
    "tg_for_eval",
]


class Mode(str, Enum):
    """Whether a read that is not confident (a guess rule, or a trace with no </think>) may produce a value."""

    REWARD = "reward"
    EVAL = "eval"


@dataclass(frozen=True)
class Parsed:
    """What was read, and on what evidence: ``value`` (the letter, the ``(start, end)`` span, or None),
    ``rule`` (the deciding rule, or the guess rule that would have decided; "" if none matched),
    ``confident`` (the rule is anchored and the trace terminated), ``in_answer_tag`` (read by the format
    policy, never by the parser), ``scope`` (answer_tag | post_think | whole_response), ``abstained``, ``reason``.
    """

    value: Union[str, Tuple[float, float], None]
    rule: str
    confident: bool
    in_answer_tag: bool
    scope: str
    abstained: bool
    reason: str = ""


def _abstain(reason: str, rule: str, in_answer_tag: bool, scope: str) -> Parsed:
    return Parsed(None, rule, False, in_answer_tag, scope, True, reason)


def _admit(value, rule: str, anchored: bool, in_answer_tag: bool, scope: str, mode: Mode) -> Parsed:
    """The parse result: confident when the rule is anchored and the trace terminated (has <answer> or
    </think>); Mode.REWARD abstains on anything not confident."""
    confident = anchored and scope != SCOPE_WHOLE
    if mode is Mode.REWARD and not confident:
        reason = "no_commitment" if anchored else "guess_rule"
        return _abstain(reason, rule, in_answer_tag, scope)
    return Parsed(value, rule, confident, in_answer_tag, scope, False)


_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

SCOPE_ANSWER_TAG = "answer_tag"
SCOPE_POST_THINK = "post_think"
SCOPE_WHOLE = "whole_response"


def select_span(response: str) -> Tuple[str, bool, str]:
    """Return ``(text, in_answer_tag, scope)``, the part of the response to read: the last ``<answer>`` block,
    else the text after the last ``</think>``, else the whole response.
    """
    blocks = list(_ANSWER_BLOCK.finditer(response or ""))
    if blocks:
        return blocks[-1].group(1), True, SCOPE_ANSWER_TAG
    if "</think>" in (response or ""):
        return response.rsplit("</think>", 1)[1], False, SCOPE_POST_THINK
    return response or "", False, SCOPE_WHOLE


# One option per line: "A. text", "A) text", "(A) text". \s* crosses a newline, so an empty "B." takes the next line.
_OPTION_LINE = re.compile(r"^\s*\(?([A-Z])[.):]\s*(.+?)\s*$", re.M)

# (rule, confident) in priority order; a rule that is not confident is a guess.
_MCQ_RULES: List[Tuple[str, bool]] = [
    ("exact", True),          # the span is nothing but an option letter
    ("phrase", True),         # an answer phrase, then a letter
    ("parentheses", True),    # (C)
    ("period", True),         # C.
    ("colon", True),          # C:
    ("right_paren", True),    # C)
    ("option_text", True),    # exactly one option's TEXT appears
    ("standalone", False),    # a lone C, not part of a word
    ("prose_start", False),   # the span begins with an option letter, then more words
    ("prose_end", False),     # the span ends with an option letter
    ("loose", False),         # a lone option letter anywhere
]
_MCQ_PRIORITY = {name: len(_MCQ_RULES) - i for i, (name, _) in enumerate(_MCQ_RULES)}
_MCQ_CONFIDENT = dict(_MCQ_RULES)

_TRIM = " \t\n\r,.!?;:'\"*"


def parse_options(question: str) -> Optional[Dict[str, str]]:
    """``{letter: option text}`` for the contiguous run of options from A, or None for fewer than two."""
    if not question:
        return None
    seen: Dict[str, str] = {}
    for letter, text in _OPTION_LINE.findall(question):
        seen.setdefault(letter, text)
    out: Dict[str, str] = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter not in seen:
            break
        out[letter] = seen[letter]
    return out if len(out) >= 2 else None


def _standalone(letter: str, text: str) -> List[int]:
    """Positions where `letter` stands alone -- never a letter inside a word."""
    return [m.start() for m in re.finditer(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", text)]


def _mcq_candidates(span: str, options: Dict[str, str]) -> List[Tuple[str, int, str]]:
    """``(letter, position, rule)`` for every rule that matches."""
    letters = list(options)
    trimmed = span.strip(_TRIM)
    padded = " " + trimmed + " "
    found: List[Tuple[str, int, str]] = []

    if trimmed in letters:
        found.append((trimmed, 0, "exact"))

    lowered = padded.lower()
    for phrase in _PHRASES:
        at = lowered.find(phrase)
        if at == -1:
            continue
        # the letter may sit inside a word ("the answer is Beta" -> B)
        for letter in letters:
            pos = padded.find(letter, at + len(phrase))
            if pos != -1:
                found.append((letter, pos, "phrase"))

    for letter in letters:
        for token, rule in ((f"({letter})", "parentheses"), (f"{letter}.", "period"),
                            (f"{letter}:", "colon"), (f"{letter})", "right_paren")):
            if token in padded:
                found.append((letter, padded.rfind(token), rule))

    # option_text: exactly one option's text appears, in a span at most twice the total option text
    body = trimmed.lower()
    budget = 2 * sum(len(str(t)) for t in options.values())
    if body and budget and len(body) <= budget:
        hits = [(letter, body.rfind(str(text).lower()))
                for letter, text in options.items()
                if str(text).lower() in body]
        if len(hits) == 1:
            found.append((hits[0][0], hits[0][1], "option_text"))

    for letter in letters:
        at = _standalone(letter, padded)
        if at:
            found.append((letter, at[-1], "standalone"))

    if len(trimmed) > 1:
        head, tail = trimmed[0], trimmed[-1]
        if head in letters and not trimmed[1].isalpha():
            found.append((head, 0, "prose_start"))
        if tail in letters and not trimmed[-2].isalpha():
            found.append((tail, len(padded) - 1, "prose_end"))

    if not found:
        for letter in letters:
            at = _standalone(letter, padded)
            if at:
                found.append((letter, at[-1], "loose"))

    return found


def parse_mcq(response: str, question: str, mode: Mode) -> Parsed:
    """Read the multiple-choice letter the model chose."""
    span, in_tag, scope = select_span(response)

    # Without parsed options, any letter A-H; free-text golds are excluded by gold_letter instead.
    options = parse_options(question) or {letter: "" for letter in _DEFAULT_CHOICES}

    found = _mcq_candidates(span, options)
    if not found:
        return _abstain("no_letter_found", "", in_tag, scope)

    # Highest-priority rule wins; within a rule, the latest position.
    letter, _, rule = max(found, key=lambda c: (_MCQ_PRIORITY[c[2]], c[1]))
    return _admit(letter, rule, _MCQ_CONFIDENT[rule], in_tag, scope, mode)


def free_text_span(response: str) -> str:
    """The answer span of `select_span`, without the provenance."""
    text, _, _ = select_span(response)
    return text


def gold_letter(gold: str, question: str) -> Optional[str]:
    """The ground-truth option letter, or None when the gold is free text."""
    options = parse_options(question)
    trimmed = (gold or "").strip(_TRIM)
    if options is None or trimmed not in options:
        return None
    return trimmed


_SEP = r"(?:\s*[-–—]\s*|\s+to\s+|\s+and\s+|\s+until\s+)"
_CLOCK = r"\d{1,2}:\d{2}(?::\d{2})?"
_PLAIN = r"\d+(?:\.\d+)?"

# Rules in priority order; within a rule the last match wins.
_TG_CLOCK_PAIR = re.compile(rf"({_CLOCK}){_SEP}({_CLOCK})", re.IGNORECASE)
_TG_PLAIN_PAIR = re.compile(rf"({_PLAIN}){_SEP}({_PLAIN})", re.IGNORECASE)
_NUMBER = re.compile(rf"(?<![\d:.]){_PLAIN}(?![\d:])")


def _seconds(token: str) -> float:
    if ":" not in token:
        return float(token)
    parts = [float(p) for p in token.split(":")]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]


def parse_tg(response: str, mode: Mode) -> Parsed:
    """Read the ``(start, end)`` interval, in seconds, that the model gave."""
    span, in_tag, scope = select_span(response)

    for pattern, rule in ((_TG_CLOCK_PAIR, "clock_pair"), (_TG_PLAIN_PAIR, "number_pair")):
        matches = pattern.findall(span)
        if matches:
            start, end = matches[-1]
            return _admit((_seconds(start), _seconds(end)), rule, True, in_tag, scope, mode)

    # last_two, a guess: the last two numbers in increasing order
    numbers = [float(m.group()) for m in _NUMBER.finditer(span)]
    for i in range(len(numbers) - 2, -1, -1):
        if numbers[i] <= numbers[i + 1]:
            return _admit((numbers[i], numbers[i + 1]), "last_two", False, in_tag, scope, mode)

    return _abstain("no_span_found", "", in_tag, scope)


def gold_span(gold) -> Optional[Tuple[float, float]]:
    """The ground-truth interval, from a pair or from the string form the eval sets use."""
    if isinstance(gold, (list, tuple)) and len(gold) >= 2:
        try:
            return float(gold[0]), float(gold[1])
        except (TypeError, ValueError):
            return None
    parsed = parse_tg(str(gold), Mode.EVAL)
    return None if parsed.abstained else parsed.value


def cal_iou_precision_recall(gt_list, pred_list):
    """Interval IoU / precision / recall of ``pred_list`` against ``gt_list``, both ``(start, end)``.

    Returns ``(-2, -2, -2)`` for an inverted ground truth and ``(-1, -1, -1)`` for a missing
    (end -1) or inverted prediction, so a caller can tell "did not parse" from a real 0.0.
    """
    start1, end1 = gt_list
    start2, end2 = pred_list
    if start1 > end1:
        print(f">> wrong annotation gt: {gt_list}, pred: {pred_list}")
        return -2, -2, -2
    elif end2 == -1 or start2 > end2:
        print(f">> err gt: {gt_list}, pred: {pred_list}")
        return -1, -1, -1
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    eps = 1e-4
    inter_length = max(0, inter_end - inter_start)
    pred_length = end2 - start2 + eps
    gt_length = end1 - start1 + eps
    union_length = pred_length + gt_length - inter_length + eps
    iou = inter_length / union_length
    precision = inter_length / pred_length
    recall = inter_length / gt_length
    return iou, precision, recall


def mcq_for_reward(response: str, question: str) -> Parsed:
    """Training reward: a read that is not confident abstains."""
    return parse_mcq(response, question, Mode.REWARD)


def mcq_for_eval(response: str, question: str) -> Parsed:
    """Evaluation: every rule is admitted."""
    return parse_mcq(response, question, Mode.EVAL)


def tg_for_reward(response: str) -> Parsed:
    """Training reward. See ``mcq_for_reward``."""
    return parse_tg(response, Mode.REWARD)


def tg_for_eval(response: str) -> Parsed:
    """Evaluation. See ``mcq_for_eval``."""
    return parse_tg(response, Mode.EVAL)

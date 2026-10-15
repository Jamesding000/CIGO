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
"""Per-row reward of the two tasks.

``score`` is 0/1 for multiple choice and the raw IoU for temporal grounding. ``pure_format`` is returned next to
it for the format gate, which penalizes a malformed row; format is never a reward term.
"""

import json
import logging
import os
from typing import Callable

from . import format_gate
from . import parse as ap


class _RewardLogFile(logging.Handler):
    """Appends records to $LOGGER_OUTPUT_FILE ("logging" -> "logging_rewards" in the path), else to
    `default`; the file is opened at the first record."""

    def __init__(self, default: str):
        super().__init__()
        self.default = default
        self.file = None
        self.setFormatter(logging.Formatter(
            "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S"))

    def emit(self, record):
        if self.file is None:
            path = os.getenv("LOGGER_OUTPUT_FILE", self.default).replace("logging", "logging_rewards")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.file = logging.FileHandler(path, mode="a")
            self.file.setFormatter(self.formatter)
        self.file.emit(record)


def _reward_logger(name: str, default: str) -> logging.Logger:
    """Level $VERL_LOGGING_LEVEL (default WARN); records go to the reward log file only."""
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
    if not any(isinstance(h, _RewardLogFile) for h in logger.handlers):
        logger.addHandler(_RewardLogFile(default))
    logger.propagate = False
    return logger


logger = _reward_logger(__name__, "/tmp/verl_reward_logs/rewards.log")


def mcq_correctness(predict_str: str, ground_truth: str, extra_info: dict) -> tuple[float, dict]:
    """1.0 when the option letter read by `parse.mcq_for_reward` is the gold letter, else 0.0."""
    parsed = ap.mcq_for_reward(predict_str, extra_info["question"])
    pred_ans = parsed.value or ""
    reward = 1.0 if pred_ans and pred_ans.strip() == ground_truth.strip() else 0.0
    log_data = {
        "reward_name": "mcq_reward",
        "mcq_reward": reward,
        "task_reward_func": reward,
        "answered": 1.0 if pred_ans else 0.0,
        "parse_rule": parsed.rule,
        "parse_scope": parsed.scope,
        "in_answer_tag": float(parsed.in_answer_tag),
    }
    return reward, log_data


def tg_correctness(predict_str: str, ground_truth: str, extra_info: dict = None) -> tuple[float, dict]:
    """The raw IoU of the span read by `parse.tg_for_reward` against the gold span; 0 when none is read."""
    if isinstance(ground_truth, str):
        ground_truth = eval(ground_truth)
    parsed = ap.tg_for_reward(predict_str)
    pred_list = parsed.value if parsed.value is not None else (-1, -1)
    # no span: the -1 sentinel scores 0
    iou = max(0, ap.cal_iou_precision_recall(ground_truth, list(pred_list))[0])
    log_data = {
        "reward_name": "iou_reward",
        "iou_reward": iou,
        "iou_raw": iou,
        "task_reward_func": iou,
        "parse_rule": parsed.rule,
        "parse_scope": parsed.scope,
        "in_answer_tag": float(parsed.in_answer_tag),
        "answered": 0.0 if parsed.value is None else 1.0,
    }
    return iou, log_data


def compose_score(predict_str: str, ground_truth: str, extra_info: dict, correctness_fn: Callable,
                  task: str) -> dict:
    """The row's reward dict: ``correctness_fn``'s value as ``score``, its log, and the format verdict.

    ``correctness_fn`` maps ``(predict_str, ground_truth, extra_info)`` to ``(value, log)``; ``task``
    ("mcq" | "tg") selects the answer payload the format gate requires.
    """
    pure_format = format_gate.pure_format_score(predict_str, task, extra_info)
    log_data = {
        "score": 0.0,
        "response": str(predict_str),
        "ground_truth": str(ground_truth),
        "format_reward": 1.0 if pure_format > 0 else 0.0,
        "pure_format": pure_format,
    }
    correctness, task_log = correctness_fn(predict_str, ground_truth, extra_info)
    log_data.update(task_log)
    log_data["score"] = float(correctness)
    log_data["total_reward_func"] = log_data["score"]
    logger.info("Compute Rewards: %s", json.dumps(log_data, ensure_ascii=False, indent=4))
    return log_data


def score_mcq(predict_str: str, ground_truth: str, extra_info: dict) -> dict:
    return compose_score(predict_str, ground_truth, extra_info, mcq_correctness, "mcq")


def score_tg(predict_str: str, ground_truth: str, extra_info: dict) -> dict:
    return compose_score(predict_str, ground_truth, extra_info, tg_correctness, "tg")

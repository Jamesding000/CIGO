"""The training reward, connected through verl's ``reward.custom_reward_function`` hook (configs/base.yaml).

parse        reads the answer out of a response (shared with evaluation) and computes interval IoU
format_gate  the tag-structure and content checks a malformed row is priced on
scorers      the per-row reward dict of each task
"""

from .scorers import score_mcq, score_tg


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info: dict) -> dict:
    """The reward dict of one response, routed by its data_source."""
    if data_source == "./data/charades_tool":
        return score_tg(solution_str, ground_truth, extra_info)
    if data_source == "./data/video_r1/multiple_choice":
        return score_mcq(solution_str, ground_truth, extra_info)
    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

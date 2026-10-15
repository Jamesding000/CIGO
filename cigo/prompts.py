"""Prompts shared by training and evaluation: THINK_GENERAL (no tool, one turn), THINK_BLIND (no video, for the
blind filter) and the tool-using TOOL_* harness."""
import json

THINK_GENERAL = """This is a video with duration {duration} seconds.
You should first think the user's question step-by-step and then provides the user with the answer. 
Note that there must be an answer for each question and you must answer it.
Output your thought process within the <think> </think> tags, and output your answer within the <answer> </answer> tags,
i.e., <think> ... </think><answer> ... </answer>.
User Question: 
{input_text}
"""

# the question without its video: the model must still commit to one answer
THINK_BLIND = """This is a video with duration {duration} seconds.
Answer the following question about the video. First think step-by-step, then commit to your single best answer — even if you are unsure, you must give one definitive answer (do not refuse or say you cannot see the video).
Output your reasoning within <think> </think> tags, then your final answer within <answer> </answer> tags, and put NOTHING else inside the answer tags.
For a multiple-choice question, the <answer> tags must contain only the letter of the chosen option, e.g. <answer>C</answer>. Otherwise, follow exactly the answer format requested in the question.
i.e., <think> ... </think><answer> ... </answer>.
User Question:
{input_text}
"""

# --- tool-using harness: system message, user message (after the overview frames), tool description, tool response

TOOL_SYSTEM = """You are a video reasoning assistant. You are given a uniformly sampled overview of a video and a question about it. You may retrieve additional frames from any time range before answering.

{tool_block}
# Procedure

Each step is one <think> </think> block followed by exactly one <tool_call> </tool_call> or one <answer> </answer>.

If the observations you currently hold do not yet show a visual detail that is essential to answering, name the time range that would contain that detail and call get_video_clip_frame on it to observe that range in fine detail.

i.e.  <think> ... </think><tool_call> ... </tool_call>

Once you believe you have observed all the video clues that help you answer the question, and are able to integrate the existing clues to give the correct answer, put the correct answer within the <answer> and </answer> tags and stop.

i.e.  <think> ... </think><answer> ... </answer>

All timestamps refer to the full video, lie within its duration, and are written to two decimal places.

The user message states a tool budget. This is the maximum number of tool calls allowed for the entire question. You must not exceed it: once the remaining budget reaches zero you must answer.

# Answer format

You must output exactly one <answer> </answer> block and it must be the last thing you produce.

For a multiple-choice question, put only the option letter inside the answer tags, e.g. <answer>C</answer>.

For a temporal grounding question, put only the start and end time in seconds inside the answer tags, e.g. <answer>The event happens in 24.30 - 30.42 seconds.</answer>.

Otherwise follow exactly the answer format requested in the question."""

TOOL_USER = """The frames above are the global context of the video: the full video uniformly sampled at {sample_fps:.2f} fps, {total_frames} frames spanning 0.00s to {max_duration:.2f}s, one frame every {frame_interval:.2f}s. An event shorter than about {frame_interval:.2f}s can fall entirely between two of these frames.

User Question:
{input_text}"""


def overview_stats(total_frames: int, sample_fps: float) -> dict:
    """The overview fields of the user prompt for `total_frames` frames actually sampled at `sample_fps`."""
    interval = 1.0 / sample_fps if sample_fps else 0.0
    return {
        "sample_fps": sample_fps,
        "total_frames": total_frames,
        "frame_interval": interval,
        "max_duration": (total_frames - 1) * interval if total_frames else 0.0,
    }


# frames returned per clip call, at most
CLIP_FRAMES = 64
TOOL_DESCRIPTION = (
    f"Inspect a time range of the video closely. Returns up to {CLIP_FRAMES} frames evenly spaced across "
    "[start_time, end_time]; the video is stored at 2 frames per second, so a range shorter than "
    f"{CLIP_FRAMES // 2} seconds returns 2 frames for each of its seconds. A narrow range therefore comes back "
    "densely sampled, while a range covering the whole video returns the same overview you "
    "already hold."
)

# prefix replay rewrites the tail after this marker in place, so both tails must tokenize to the same length
REMAINING_MARKER = "Next step:"

# with calls left both actions are named, so the sentence does not nudge toward calling
_REMAINING_MORE = "You may answer or call another tool in this round."
_REMAINING_LAST = "You must answer in this round; no calls remain."


def remaining_tail(k: int) -> str:
    """The text after REMAINING_MARKER for a row with `k` tool calls left (only whether calls remain, not the count)."""
    return " " + (_REMAINING_LAST if int(k) <= 0 else _REMAINING_MORE)


# --- tool-response envelope: observation index, windows held, whether calls remain


def clamp_span(start_time: float, end_time: float, duration: float) -> tuple[float, float]:
    """Clamp a requested clip span to ``[0, duration]``; reversed spans are swapped and spans under 1 s widened."""
    if end_time < start_time:
        start_time, end_time = end_time, start_time
    start_time = min(max(start_time, 0.0), duration)
    end_time = min(max(end_time, 0.0), duration)
    min_len = min(1.0, duration)
    if end_time - start_time < min_len:
        end_time = min(start_time + min_len, duration)
        start_time = max(0.0, end_time - min_len)
    return start_time, end_time


def call_windows(calls) -> list[tuple[float, float]]:
    """`[(start, end), ...]` requested by the clip calls of a call log; entries without a usable window are skipped.

    Accepts flat entries (``{"start_time", "end_time", ...}``) and nested ones (``{"name", "arguments"}``,
    arguments possibly a JSON string).
    """
    out = []
    for c in calls or []:
        a = c
        if isinstance(c, dict) and "arguments" in c:
            a = c["arguments"]
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    continue
        if not isinstance(a, dict):
            continue
        try:
            out.append((float(a["start_time"]), float(a["end_time"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def envelope_kwargs(prior_calls, budget: int, duration: float | None = None,
                    turns_remaining: int | None = None) -> dict:
    """The tool-response fields for a call made after `prior_calls` (which excludes the current call).

    `budget` is the row's total call allowance. Returns `call_index` (1-based), `retrieved_windows` (clamped
    to `duration` when given) and `budget_remaining` after this call, capped by `turns_remaining` when given.
    """
    prior = call_windows(prior_calls)
    if duration is not None:
        # the windows the tool actually returned
        prior = [clamp_span(s, e, float(duration)) for s, e in prior]
    spent = len(prior) + 1
    left = int(budget) - spent
    if turns_remaining is not None:
        left = min(left, int(turns_remaining))
    return {
        "call_index": spent,
        "retrieved_windows": prior,
        "budget_remaining": max(0, left),
    }


TOOL_RESPONSE = (
    "Fine-grained Observation {obs_idx} - {start_time:.2f}s to {end_time:.2f}s, "
    "{n_frames} frames, one every {clip_interval:.2f}s.\n"
    "Retrieved so far: {retrieved_windows}\n"
    + REMAINING_MARKER + "{tail}"
)

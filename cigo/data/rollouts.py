"""Sample N answers per question with vLLM and score each with the training reward (shared by the two filters).

A question file is a JSON list of rows {id, video, text, solution, duration}; `solution` is an option letter for a
multiple-choice question and [start, end] in seconds for a temporal-grounding one. Rollouts are appended to
<work-dir>/rollouts_<shard>.jsonl, so an interrupted run resumes where it stopped.
"""

import argparse
import json
import os
import subprocess
import sys

from cigo import prompts
from cigo.reward.scorers import mcq_correctness, tg_correctness

SYSTEM = "You are a helpful assistant."
STOP = ["</answer>", " </answer>"]


def is_grounding(row: dict) -> bool:
    return isinstance(row["solution"], (list, tuple))


def score(row: dict, response: str) -> float:
    """Correctness of one response: 1/0 for multiple choice, IoU with the gold interval for grounding."""
    if is_grounding(row):
        return tg_correctness(response, list(row["solution"]))[0]
    return mcq_correctness(response, row["solution"], {"question": row["text"]})[0]


def add_args(parser: argparse.ArgumentParser, n: int, temperature: float) -> None:
    parser.add_argument("--model", required=True, help="model that answers, e.g. Qwen3-VL-8B-Thinking")
    parser.add_argument("--input", required=True, help="question file (JSON list)")
    parser.add_argument("--output", required=True, help="kept questions (JSON list); <output>.stats.json per row")
    parser.add_argument("--work-dir", required=True, help="rollouts, resumable")
    parser.add_argument("--n", type=int, default=n, help="answers per question")
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-gpus", type=int, default=1, help="one vLLM engine per GPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)


def load_rows(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def _video_input(row, frame_dir, max_frames, max_pixels):
    """(prompt fields, vLLM video item) of a row's overview frames; None when its frames are missing."""
    from qwen_vl_utils import fetch_video

    from verl.utils.dataset.rl_dataset_video_tool import resolve_and_sample_frames

    sampled = resolve_and_sample_frames(os.path.join(frame_dir, row["video"]), max_frames)
    if sampled is None:
        return None
    frames, fps = sampled
    ele = {"type": "video", "video": frames, "sample_fps": fps, "min_pixels": 4 * 28 * 28, "max_pixels": max_pixels}
    video, _ = fetch_video(ele, return_video_sample_fps=True, return_video_metadata=True)
    return prompts.overview_stats(len(frames), fps), video


def _request(processor, row, video_args):
    """vLLM request of one row: the question alone (blind), or after its overview frames."""
    duration = round(float(row.get("duration") or 0.0), 2)
    if video_args is None:
        text = prompts.THINK_BLIND.format(input_text=row["text"], duration=duration)
        content = [{"type": "text", "text": text}]
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        return {"prompt": processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}
    video = _video_input(row, *video_args)
    if video is None:
        return None
    stats, item = video
    text = prompts.THINK_GENERAL.format(input_text=row["text"], duration=duration, **stats)
    content = [{"type": "video"}, {"type": "text", "text": text}]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"video": [item]}}


def _sample_shard(args, video_args, batch_size=32):
    """Generate and score this shard's questions, appending one line per question."""
    import vllm
    from transformers import AutoProcessor

    rows = load_rows(args.input)[args.shard :: args.num_gpus]
    path = os.path.join(args.work_dir, f"rollouts_{args.shard}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(line)["id"] for line in f}
        rows = [r for r in rows if r["id"] not in done]
    if not rows:
        return

    llm = vllm.LLM(
        model=args.model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        disable_mm_preprocessor_cache=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    params = vllm.SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=STOP,
        include_stop_str_in_output=True,
    )
    for i in range(0, len(rows), batch_size):
        batch = [(r, _request(processor, r, video_args)) for r in rows[i : i + batch_size]]
        missing = [r["id"] for r, req in batch if req is None]
        if missing:
            print(f"skipped {len(missing)} questions without frames: {missing[:5]}", flush=True)
        batch = [(r, req) for r, req in batch if req is not None]
        outputs = llm.generate([req for _, req in batch], params, use_tqdm=False)
        with open(path, "a") as f:
            for (row, _), out in zip(batch, outputs, strict=True):
                texts = [o.text for o in out.outputs]
                line = {"id": row["id"], "responses": texts, "scores": [score(row, t) for t in texts]}
                f.write(json.dumps(line) + "\n")
        print(f"[shard {args.shard}] {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)


def sample(args, video_args=None) -> dict:
    """{id: scores} of every question, sampling the ones not yet in the work dir (one process per GPU)."""
    os.makedirs(args.work_dir, exist_ok=True)
    if args.shard is not None:
        _sample_shard(args, video_args)
        return {}
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.num_gpus)))).split(",")
    workers = [
        subprocess.Popen(
            [sys.executable, "-m", args.module, *sys.argv[1:], "--shard", str(k)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus[k]},
        )
        for k in range(args.num_gpus)
    ]
    if any(w.wait() for w in workers):
        sys.exit("a sampling worker failed")
    scores = {}
    for k in range(args.num_gpus):
        with open(os.path.join(args.work_dir, f"rollouts_{k}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                scores[r["id"]] = r["scores"]
    return scores


def write(args, rows: list, stats: dict, keep) -> None:
    """Write the kept rows and every row's statistics; print the counts."""
    kept = [r for r in rows if r["id"] in stats and keep(stats[r["id"]])]
    with open(args.output, "w") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    with open(os.path.splitext(args.output)[0] + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"{len(kept)} of {len(rows)} questions kept ({len(rows) - len(stats)} without rollouts) -> {args.output}")

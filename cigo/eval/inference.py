"""Multi-turn video QA inference for Qwen3-VL with vLLM, using the training prompts and video tool.

Messages carry bare `{"type": "video"}` placeholders; each row's videos (the overview, then every retrieved
clip) are passed to vLLM as `multi_modal_data["video"]`.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch
import vllm
from qwen_vl_utils import fetch_video
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor

from cigo import prompts
from cigo.reward import format_gate
from verl.utils.dataset.rl_dataset_video_tool import (
    build_system_prompt,
    resolve_and_sample_frames,
    resolve_frame_dir,
    tools_and_system_prompt,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", 0.6))
EVAL_TOP_P = float(os.environ.get("EVAL_TOP_P", 1.0))
EVAL_REPETITION_PENALTY = float(os.environ.get("EVAL_REPETITION_PENALTY", 1.05))
# Per-turn token cap; must equal training's single_turn_response_length.
MAX_RESPONSE_TOKENS = int(os.environ.get("MAX_RESPONSE_TOKENS", 4096))
# Total assistant generations; must be >= tool budget + 1.
MAX_ASSISTANT_TURNS = os.environ.get("MAX_ASSISTANT_TURNS")
EVAL_TOOL_BUDGET = int(os.environ.get("EVAL_TOOL_BUDGET", 3))

# A row still unanswered at the turn cap gets one more generation after FORCE_ANSWER_TEXT.
EVAL_FORCE_ANSWER = os.environ.get("EVAL_FORCE_ANSWER", "1") == "1"
FORCE_ANSWER_TEXT = (
    "Continue your reasoning process inside <think> and </think>. You have reached the final "
    "turn and MUST provide the final answer based on all the previous conversation and "
    "observations. Think carefully about all the information you have gathered inside <think> "
    "and </think> and give your best answer inside <answer> and </answer>. Do NOT call the "
    "tool again."
)


_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_output(text: str) -> dict:
    m = _TOOL_RE.search(text)
    if m:
        # Executes any call training's hermes parser would; the stricter format rule is only recorded.
        try:
            payload = json.loads(m.group(1))
            assert "name" in payload and "arguments" in payload
            assert isinstance(payload, dict) and isinstance(payload["name"], str)
            return {"type": "tool", "content": payload}
        except Exception as e:
            return {"type": "error", "content": str(e)}
    m = _ANSWER_RE.search(text)
    if m:
        return {"type": "answer", "content": m.group(1)}
    return {"type": "error", "content": "Error: response parse failed, output text: " + text[-500:]}


def close_unclosed_think(text: str) -> str:
    """Append </think> to a turn that opened a think and was cut before closing it."""
    if "<think>" in text and "</think>" not in text:
        return text + "</think>"
    return text


def committed(text: str) -> bool:
    """True when the turn holds a non-empty <answer>, which ends the episode."""
    m = _ANSWER_RE.search(text or "")
    return m is not None and bool(m.group(1).strip())


def end_turn_at_think(text: str) -> str:
    """Cut the turn at its last </think> (closing an unclosed think first), dropping an unexecuted final call."""
    text = close_unclosed_think(text)
    idx = text.rfind("</think>")
    if idx == -1:
        return text
    return text[: idx + len("</think>")]


@dataclass
class _AgentDataShim:
    """Stand-in for verl `AgentData`: the tool reads `tools_kwargs` from it."""

    tools_kwargs: dict


def execute_tool_sync(tool, parameters: dict, tools_kwargs: dict):
    """Run a verl async BaseTool from sync code."""
    response, _, _ = asyncio.run(
        tool.execute(
            instance_id=str(uuid.uuid4()),
            parameters=parameters,
            agent_data=_AgentDataShim(tools_kwargs=tools_kwargs),
        )
    )
    return response


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pending_items(args):
    """This chunk's items of the ground-truth file that the answers file does not hold yet."""
    with open(args.gt_file) as f:
        data = json.load(f)
    data = data[args.chunk_idx::args.num_chunks]

    if os.path.exists(args.answers_file):
        with open(args.answers_file) as f:
            seen = {json.loads(line)["id"] for line in f}
        data = [s for s in data if s["id"] not in seen]
    return data


def _tools_and_system_prompt(use_tool, tool_config):
    """({name: tool}, system prompt): the training tool and prompt with --use_tool, else no tool."""
    if not use_tool:
        return {}, build_system_prompt([], template=None)
    return tools_and_system_prompt(tool_config or os.path.join(_REPO, "verl/verl/tools/config/video_tool_config.yaml"))


class VideoQADataset(Dataset):
    """Rows with their `messages` (bare video placeholders) and decoded overview `videos_init = [(tensor, metadata)]`."""

    def __init__(self, args):
        data = _pending_items(args)

        self.use_tool = bool(args.use_tool)
        self.template = prompts.TOOL_USER if self.use_tool else getattr(prompts, args.prompt_type)
        self.max_frames = args.max_frames
        self.max_pixels = args.max_pixels
        self.tools, self.system_prompt = _tools_and_system_prompt(self.use_tool, args.tool_config)

        # Rows without a frame directory are skipped, as in training.
        self.data = []
        for sample in data:
            video_path = os.path.join(args.video_dir, sample["video"])
            if resolve_frame_dir(video_path) is None:
                logger.warning(f"Frame directory not found, skipping: {video_path}")
                continue
            sample["video_path"] = video_path
            self.data.append(sample)
        print(f">>> prompt: {'tool' if self.use_tool else args.prompt_type}  (use_tool={self.use_tool})")
        print(f">>> samples: {len(self.data)} (skipped {len(data) - len(self.data)})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = dict(self.data[idx])
        sample.setdefault("question", sample.get("text"))
        sample.setdefault("answer", str(sample.get("solution", "")))

        result = resolve_and_sample_frames(sample["video_path"], self.max_frames)
        assert result is not None, f"Empty frame dir for {sample['video_path']}"
        frame_uris, sample_fps = result
        # Without a recorded duration, the frames span len(frame_uris) / sample_fps seconds.
        sample["duration"] = sample.get("duration") or len(frame_uris) / sample_fps

        ele = {
            "type": "video",
            "video": frame_uris,
            "sample_fps": sample_fps,
            "min_pixels": 4 * 28 * 28,
            "max_pixels": self.max_pixels,
        }
        (tensor, metadata), _ = fetch_video(
            ele, return_video_sample_fps=True, return_video_metadata=True
        )
        sample["videos_init"] = [(tensor, metadata)]

        duration = round(float(sample["duration"]), 2)
        question = self.template.format(input_text=sample["question"], duration=duration,
                                        **prompts.overview_stats(len(frame_uris), sample_fps))
        sample["messages"] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        return sample


# Stop after a closing </tool_call> or </answer>.
STOP = [t.format(tag=tag) for tag in ("tool_call", "answer")
        for t in ("</{tag}>", " </{tag}>", "</{tag}>\n", " </{tag}>\n", "</{tag}>\n\n", " </{tag}>\n\n")]


@dataclass
class _Episode:
    """One row's conversation and counters during the rollout."""

    sample: dict
    messages: list
    videos: list  # (tensor, metadata) of the overview and of every retrieved clip
    pred: str = ""  # every generated turn, concatenated
    last_turn: str = ""
    active: bool = True
    valid_action: int = 0
    valid_tool_call: int = 0
    # Whether training's format gate would accept every decoded call; reported only.
    tool_calls_valid: bool = True
    valid_tool_exec: int = 0
    tool_calls: list = field(default_factory=list)
    forced: bool = False
    pruned: bool = False


def _build_vllm_input(processor, messages, videos):
    """The vLLM request of a conversation, re-rendered from scratch each turn."""
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"video": videos}}


def _start_engine(args):
    """(vLLM engine, processor, batch size), sized by the GPU memory."""
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        card = f"{vram_gb:.0f} GiB card"
    except Exception as err:
        vram_gb, card = 0.0, f"card probe failed ({err})"
    big_card = vram_gb >= 80
    # Avoids fragmentation OOMs on 48 GB cards; read by the engine process.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    llm = vllm.LLM(
        model=args.model_path,
        dtype="bfloat16",
        seed=args.seed,
        max_num_batched_tokens=8192,
        disable_log_stats=True,
        max_model_len=65536,
        gpu_memory_utilization=0.7 if big_card else 0.9,
        enforce_eager=True,
        disable_mm_preprocessor_cache=True,  # vLLM 0.11's cache asserts on large video items
    )
    # Only renders the chat template; vLLM tokenizes and preprocesses the video itself.
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    print(f"[cuda:{args.chunk_idx}] Load model and processor success!")

    # The batch size changes individual predictions, as a different seed would.
    default_bs = 32 if big_card else 8
    override = os.environ.get("EVAL_BATCH_SIZE")
    batch_size = max(1, int(override or default_bs))
    print(f"[cuda:{args.chunk_idx}] eval batch_size={batch_size} "
          f"({card} -> default {default_bs}"
          f"{'' if override is None else f', EVAL_BATCH_SIZE={override} overrides'})", flush=True)
    return llm, processor, batch_size


def _rollout(llm, processor, sampling_params, dataset, episodes, max_turns):
    """Up to `max_turns` generations per row: each turn commits an answer, calls the tool, or neither."""
    for step in range(max_turns):
        live = [e for e in episodes if e.active]
        if not live:
            break

        # A tool call on the last generation is not executed.
        final_step = step == max_turns - 1
        inputs = [_build_vllm_input(processor, e.messages, e.videos) for e in live]
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)

        for e, output in zip(live, outputs):
            response_text = output.outputs[0].text
            if output.outputs[0].finish_reason == "length":
                response_text = close_unclosed_think(response_text)
            e.last_turn = response_text
            e.pred += response_text
            parsed = parse_output(response_text)
            e.messages.append({"role": "assistant", "content": response_text})

            if parsed["type"] == "answer":
                if committed(response_text):
                    e.active = False
                    e.valid_action += 1
                continue
            if parsed["type"] != "tool":
                continue

            e.valid_action += 1
            e.valid_tool_call += 1
            if not format_gate.tool_calls_parseable(response_text):
                e.tool_calls_valid = False
            if final_step:
                continue
            # Calls still executable after this one: generations step+1 .. max_turns-2.
            _call_tool(dataset, e, parsed["content"], turns_remaining=max_turns - step - 2)


def _call_tool(dataset, e, tool_call, turns_remaining):
    """Run one tool call of the row: append the tool message and the call-log entry."""
    tool = dataset.tools.get(tool_call["name"])
    if tool is None:
        err = f"Error: unknown tool '{tool_call['name']}'"
        e.messages.append({"role": "tool", "content": err})
        e.tool_calls.append({"error": err})
        return

    duration = float(e.sample["duration"])
    tools_kwargs = {
        "video_path": e.sample["video_path"],
        "duration": duration,
        **prompts.envelope_kwargs(e.tool_calls, budget=EVAL_TOOL_BUDGET, duration=duration,
                                  turns_remaining=turns_remaining),
    }
    args_dict = tool_call["arguments"] or {}
    # String `arguments` go to the tool as-is, which returns an error, as in training.
    st, et = ((args_dict.get("start_time"), args_dict.get("end_time"))
              if isinstance(args_dict, dict) else (None, None))
    tool_response = execute_tool_sync(tool, args_dict, tools_kwargs)

    if tool_response.video:
        e.videos.extend(tool_response.video)
        e.valid_tool_exec += 1
        tool_tensor, _ = tool_response.video[0]
        n_frames = int(tool_tensor.shape[0])
        e.tool_calls.append({"start_time": st, "end_time": et, "n_frames": n_frames})
        e.messages.append({
            "role": "tool",
            "content": [
                {"type": "video"},
                {"type": "text", "text": tool_response.text or ""},
            ],
        })
    else:
        err_text = (tool_response.text or "").replace(chr(10), " ")
        e.tool_calls.append({
            "start_time": st, "end_time": et, "error": err_text,
        })
        e.messages.append({"role": "tool", "content": err_text})


def _force_answer(llm, processor, sampling_params, episodes):
    """Give every unanswered row one more generation after an instruction to answer."""
    todo = [e for e in episodes if e.active]
    if not todo:
        return
    for e in todo:
        tidy = end_turn_at_think(e.last_turn)
        if tidy != e.last_turn:
            e.pruned = True
            if e.pred.endswith(e.last_turn):
                e.pred = e.pred[: len(e.pred) - len(e.last_turn)] + tidy
            for msg in reversed(e.messages):
                if msg["role"] == "assistant":
                    msg["content"] = tidy
                    break
        e.messages.append({"role": "user", "content": FORCE_ANSWER_TEXT})
    f_params = sampling_params.clone()
    f_params.stop = [t for t in sampling_params.stop if "tool_call" not in t]
    outs = llm.generate([_build_vllm_input(processor, e.messages, e.videos) for e in todo], f_params, use_tqdm=False)
    for e, out in zip(todo, outs):
        txt = out.outputs[0].text
        if out.outputs[0].finish_reason == "length":
            txt = close_unclosed_think(txt)
        e.pred += txt
        e.messages.append({"role": "assistant", "content": txt})
        e.forced = True
        if committed(txt):
            e.active = False


def _row(processor, e):
    """The prediction row written for one episode."""
    row = {
        "id": e.sample["id"],
        "question": e.sample["question"],
        "answer": e.sample["answer"],
        "pred": e.pred,
        "text": processor.apply_chat_template(e.messages, tokenize=False, add_generation_prompt=False),
        "video_path": e.sample["video_path"],
        "tool_calls": e.tool_calls,
        "meta_info": {
            "forced_answer": e.forced,
            "forced_answer_pruned": e.pruned,
            "active_mask": e.active,
            "valid_action_stats": e.valid_action,
            "valid_tool_call_stats": e.valid_tool_call,
            "tool_calls_valid": e.tool_calls_valid,
            "valid_tool_exec_stats": e.valid_tool_exec,
        },
    }
    if "question_type" in e.sample:
        row["question_type"] = e.sample["question_type"]
    return row


def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = (
        f"{args.num_chunks}_{args.chunk_idx}" if args.num_chunks > 1 else args.output_name
    )
    args.answers_file = os.path.join(args.output_dir, f"{output_name}.json")

    dataset = VideoQADataset(args)
    if len(dataset) == 0:
        # Nothing left: skip the engine, but leave a (possibly empty) shard for the scorer.
        open(args.answers_file, "a").close()
        print(f"[cuda:{args.chunk_idx}] nothing left to run in {args.answers_file}", flush=True)
        return

    llm, processor, batch_size = _start_engine(args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(os.environ.get("EVAL_NUM_WORKERS", 4)),
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda batch: batch,
    )
    sampling_params = vllm.SamplingParams(
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        repetition_penalty=EVAL_REPETITION_PENALTY,
        max_tokens=MAX_RESPONSE_TOKENS,
        stop=STOP,
        include_stop_str_in_output=True,
    )
    print(f"[cuda:{args.chunk_idx}] sampling: T={sampling_params.temperature} top_p={sampling_params.top_p} "
          f"rep={sampling_params.repetition_penalty} "
          f"max_tokens={sampling_params.max_tokens}", flush=True)

    default_turns = EVAL_TOOL_BUDGET + 1 if dataset.use_tool else 1
    max_turns = int(MAX_ASSISTANT_TURNS) if MAX_ASSISTANT_TURNS is not None else default_turns
    print(f"[cuda:{args.chunk_idx}] assistant turns={max_turns}  tool_budget={EVAL_TOOL_BUDGET}",
          flush=True)
    restart_every = int(os.environ.get("EVAL_RESTART_EVERY", 0))
    with open(args.answers_file, "a") as ans_file:
        for batch_idx, samples in enumerate(tqdm(dataloader, desc=f"cuda:{args.chunk_idx}")):
            if restart_every and batch_idx >= restart_every:
                # Exit 75: the launcher reruns the chunk in a fresh process, which resumes after the finished rows.
                raise SystemExit(75)
            episodes = [_Episode(s, s["messages"], list(s["videos_init"])) for s in samples]
            _rollout(llm, processor, sampling_params, dataset, episodes, max_turns)
            if EVAL_FORCE_ANSWER:
                _force_answer(llm, processor, sampling_params, episodes)
            for e in episodes:
                ans_file.write(json.dumps(_row(processor, e)) + "\n")
                ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--prompt_type", type=str, default="THINK_GENERAL",
                        help="User prompt without the tool, a template name in cigo/prompts.py")
    parser.add_argument("--use_tool", action="store_true",
                        help="Evaluate with the video tool and the prompts it was trained with")
    parser.add_argument("--tool_config", type=str, default=None,
                        help="Tool YAML; must match the one the checkpoint was trained with")

    args = parser.parse_args()
    args.seed = 42

    set_seed(args.seed)
    run_inference(args)

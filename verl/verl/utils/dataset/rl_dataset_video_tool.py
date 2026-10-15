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
import copy
import json
import logging
import os
from typing import Optional

import torch
from cigo import prompts  # prompt templates shared with the evaluation harness
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)

DEFAULT_FPS = 2.0  # the rate the frames were extracted at


def build_system_prompt(tool_schemas, template: str | None = None):
    """System message carrying the tool schemas in exactly one `# Tools` block, for training and evaluation.

    ``template`` is a whole system message with a ``{tool_block}`` placeholder; unset, the block follows
    "You are a helpful assistant."
    """
    sys_prompt = "You are a helpful assistant."
    if not tool_schemas:
        return template.replace("{tool_block}", "") if template else sys_prompt
    tool_schemas_str = "\n".join(json.dumps(x) for x in tool_schemas)
    tool_call_prompt = f"""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""
    block = tool_call_prompt
    if template is not None:
        return template.replace("{tool_block}", block.strip("\n"))
    return sys_prompt + "\n" + block


def tools_and_system_prompt(tool_config_path: str | None) -> tuple[dict, str]:
    """({name: tool}, system message) for the tools of `tool_config_path`, shared by training and eval.

    The tool description the model reads is cigo.prompts.TOOL_DESCRIPTION, not the YAML's.
    """
    from verl.tools.utils.tool_registry import initialize_tools_from_config

    tools = {t.name: t for t in initialize_tools_from_config(tool_config_path)} if tool_config_path else {}
    schemas = [t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in tools.values()]
    if schemas:
        schemas[0]["function"]["description"] = prompts.TOOL_DESCRIPTION
    return tools, build_system_prompt(schemas, template=prompts.TOOL_SYSTEM)


def resolve_frame_dir(video_path: str) -> str | None:
    """The pre-extracted frame directory of a video: its path cut at the first dot of the
    basename. None when the directory does not exist."""
    frame_dir = os.path.join(
        os.path.dirname(video_path),
        os.path.basename(video_path).split(".", 1)[0],
    )
    return frame_dir if os.path.isdir(frame_dir) else None


def list_frames(video_path: str) -> list[str] | None:
    """Paths of the pre-extracted frames of a video in frame order, or None when its
    frame directory does not exist."""
    frame_dir = resolve_frame_dir(video_path)
    if frame_dir is None:
        return None
    frame_names = os.listdir(frame_dir)
    frame_names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return [os.path.join(frame_dir, f) for f in frame_names]


def evenly_spaced(items: list, n: int) -> list:
    """`n` of `items`, evenly spaced from the first to the last."""
    idx = torch.linspace(0, len(items) - 1, n).round().long().tolist()
    return [items[i] for i in idx]


def resolve_and_sample_frames(video_path, max_frames=None, original_fps=DEFAULT_FPS):
    """``(frame_uris, sample_fps)`` for a video's frames, evenly sub-sampled to *max_frames*;
    None when the frame directory is missing or empty."""
    frame_paths = list_frames(video_path)
    if not frame_paths:
        return None

    total = len(frame_paths)
    if max_frames and total > max_frames:
        frame_paths = evenly_spaced(frame_paths, max_frames)
        sample_fps = max_frames * original_fps / total
    else:
        sample_fps = original_fps

    frame_uris = [f"file://{p}" for p in frame_paths]
    return frame_uris, sample_fps


class RLHFDatasetVideoTool(Dataset):
    """Lightweight dataset for video tool-use RL training.

    Reads JSON data files and produces the fields expected by the async
    agent loop (``raw_prompt``, ``tools_kwargs``, ``extra_info``, etc.).
    All tokenization / vision processing is deferred to the agent loop.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = list(data_files)
        self.max_samples = max_samples

        self.video_kwargs = dict(config.multi_turn.video_kwargs)

        tools, self.system_prompt = tools_and_system_prompt(config.get("tool_config_path", None))
        # Exactly one `# Tools` block: the chat template adds a second if `tools=` reaches it
        want = 1 if tools else 0
        got = self.system_prompt.count("# Tools")
        assert got == want, (
            f"system prompt renders '# Tools' {got} times, expected {want}. "
            "Something other than build_system_prompt is writing the tool block "
            "(check for `tools=` reaching apply_chat_template, and the processor's "
            "chat template version)."
        )

        self.user_template = prompts.TOOL_USER

        video_bases = config.multi_turn.video_base
        if not isinstance(video_bases, (list, ListConfig)):
            video_bases = [video_bases]
        self.video_bases = video_bases

        self._read_files()

    def _read_files(self):
        dataframe_list = []
        idx = 0
        # One video root per data file, or a single root shared by all of them
        if len(self.video_bases) == 1:
            video_bases = list(self.video_bases) * len(self.data_files)
        elif len(self.video_bases) == len(self.data_files):
            video_bases = list(self.video_bases)
        else:
            raise ValueError(
                f"multi_turn.video_base has {len(self.video_bases)} entries but there are "
                f"{len(self.data_files)} data files: {self.data_files}. Pass one root per file "
                f"(or exactly one root for all of them)."
            )
        for json_file, video_base in zip(self.data_files, video_bases, strict=True):
            raw_data = json.load(open(json_file))
            for item in raw_data:
                idx += 1
                row = self._make_row(item, idx, str(video_base))
                if row is not None:
                    dataframe_list.append(row)
        logger.info(f"Dataset loaded: {len(dataframe_list)} samples")

        if self.max_samples > 0 and self.max_samples < len(dataframe_list):
            dataframe_list = dataframe_list[: self.max_samples]
        self.dataframe = dataframe_list

    def _make_row(self, example, idx, video_base):
        duration = round(float(example["duration"]), 2)
        video_path = os.path.join(video_base, example["video"])
        video_content = {
            "type": "video",
            "video": video_path,
            "min_pixels": 4 * 28 * 28,
            "max_pixels": 224 * 224,
            "max_frames": 64,
        }
        # A row's own video_kwargs override the config's
        video_content.update(self.video_kwargs)
        if isinstance(example.get("video_kwargs"), dict):
            video_content.update(example["video_kwargs"])

        # The user prompt describes the overview frames, so sample them exactly as process_vision_info will
        sampled = resolve_and_sample_frames(video_path, int(video_content.get("max_frames", 64)))
        if sampled is None:
            logger.warning(f"No frames for {video_path}, skipping")
            return None
        frame_uris, sample_fps = sampled
        prompt = self.user_template.format(
            input_text=example["text"], duration=duration, **prompts.overview_stats(len(frame_uris), sample_fps)
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [video_content, {"type": "text", "text": prompt}]},
        ]

        return {
            "raw_prompt": messages,
            "data_source": "./data/" + example["data_source"],
            "reward_model": {"style": "rule", "ground_truth": example["solution"]},
            "extra_info": {
                "index": idx,
                "question": example["text"],
                # Seed corpus: data_source does not distinguish the multiple-choice corpora
                "corpus": example.get("corpus"),
            },
            "tools_kwargs": {
                "video_path": video_path,
                "duration": duration,
            },
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict = dict(self.dataframe[item])

        # DataProto needs at least one tensor field.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
        return row_dict

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size: int = 14,
        config: DictConfig = None,
    ) -> tuple[list, list]:
        """Replace each video path with its sub-sampled frame list and effective
        ``sample_fps``, then delegate to ``qwen_vl_utils.process_vision_info``, which then does
        no further temporal sampling."""
        messages = copy.deepcopy(messages)
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "video":
                    continue
                video_val = item.get("video")
                if not isinstance(video_val, str):
                    continue
                max_frames = item.pop("max_frames", 64)
                result = resolve_and_sample_frames(video_val, max_frames)
                if result is None:
                    raise ValueError(f"Frame directory not found for {video_val}")
                frame_uris, sample_fps = result
                item["video"] = frame_uris
                item["sample_fps"] = sample_fps

        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

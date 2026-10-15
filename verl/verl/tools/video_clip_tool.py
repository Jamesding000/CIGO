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
import logging
import math
import os
from typing import Any

from cigo import prompts

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse
from verl.utils.dataset.rl_dataset_video_tool import DEFAULT_FPS, evenly_spaced, list_frames

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetVideoClipFrameTool(BaseTool):
    """Video temporal zoom-in tool that retrieves frames from a sub-clip.

    Returns a ``ToolResponse`` with ``video`` containing ``[(frames_tensor, metadata)]``, the format the
    agent loop's vision pipeline expects.
    """

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        try:
            # Per-sample video_path, duration and envelope fields come from agent_data.tools_kwargs.
            tools_kwargs = agent_data.tools_kwargs if agent_data else kwargs
            duration = float(tools_kwargs.get("duration"))
            start_time = float(parameters.get("start_time"))
            end_time = float(parameters.get("end_time"))
            start_time, end_time = prompts.clamp_span(start_time, end_time, duration)

            frame_paths, sample_fps = self._window_frames(
                tools_kwargs.get("video_path"), start_time, end_time, prompts.CLIP_FRAMES
            )
            video, video_metadata = self._fetch_clip(frame_paths, sample_fps, start_time)
            text = self._render_response(tools_kwargs, start_time, end_time, len(frame_paths))
            return (
                ToolResponse(
                    video=[(video, video_metadata)],
                    text=text,
                ),
                0.0,
                {},
            )
        except Exception as e:
            logger.error(f"[GetVideoClipFrameTool] Execution failed: {e}, parameters: {parameters}")
            return (
                ToolResponse(text=f"Error: {self.name} tool execution failed: {e}"),
                0.0,
                {"error": str(e)},
            )

    def _window_frames(self, video_path, start_time: float, end_time: float, clip_frames: int):
        """The frames of [start_time, end_time]: `clip_frames` of them evenly spaced across the window, whatever
        its width, so a narrow window comes back dense and a wide one no denser than the overview (under a fixed
        fps a wider window would return more frames). Returns (frame paths, their sampling fps)."""
        frame_paths = list_frames(video_path)
        fps = self.config.get("fps", DEFAULT_FPS)
        total = len(frame_paths)
        assert total > 0, f"no extracted frames for {video_path}"
        # The extractor caps the frame count, so clamp in frame space too, keeping at least one frame
        start_frame = min(max(math.ceil(start_time * fps), 0), total - 1)
        end_frame = min(max(math.floor(end_time * fps), start_frame), total - 1)
        frame_paths = frame_paths[start_frame : end_frame + 1]

        n_keep = min(clip_frames, len(frame_paths))
        frame_paths = evenly_spaced(frame_paths, n_keep)
        span = max(end_time - start_time, 1e-6)
        sample_fps = (n_keep - 1) / span if n_keep > 1 else fps
        return frame_paths, sample_fps

    def _fetch_clip(self, frame_paths: list[str], sample_fps: float, start_time: float):
        """Load the frames as a video (qwen_vl_utils frame-list path) with absolute times: drawn on each frame
        and in the timestamp tokens the processor writes."""
        from qwen_vl_utils import fetch_video
        from qwen_vl_utils.vision_process import fetch_image

        from verl.utils.dataset.frame_timestamps import draw_timestamps

        frame_uris = [f"file://{p}" for p in frame_paths]
        px = {
            "min_pixels": self.config.get("min_pixels", 4 * 28 * 28),
            "max_pixels": self.config.get("max_pixels", 64 * 28 * 28),
        }
        # Absolute times are drawn here, after the resize (qwen_vl_utils would stamp clip frame k at k/sample_fps)
        images = [fetch_image({"image": u, **px}) for u in frame_uris]
        draw_timestamps(images, sample_fps, start_time=start_time)

        ele = {"type": "video", "video": images, "sample_fps": sample_fps, **px}
        (video, video_metadata), _ = fetch_video(ele, return_video_sample_fps=True, return_video_metadata=True)
        # Shift frames_indices so the processor's timestamp tokens read start_time + k/fps
        offset = start_time * video_metadata["fps"]
        video_metadata["frames_indices"] = [i + offset for i in video_metadata["frames_indices"]]
        return video, video_metadata

    @staticmethod
    def _render_response(tools_kwargs: dict, start_time: float, end_time: float, n_kept: int) -> str:
        """The tool-response envelope: observation index, window, sampling interval of these frames, every
        window retrieved so far and whether calls remain. Index, history and budget come from the agent loop
        via tools_kwargs; the defaults serve a loop that sets none."""
        clip_interval = (end_time - start_time) / (n_kept - 1) if n_kept > 1 else 0.0
        obs_idx = int(tools_kwargs.get("call_index", 1))
        k = int(tools_kwargs.get("budget_remaining", 0))
        prior = tools_kwargs.get("retrieved_windows") or []
        spans = [f"{w[0]:.2f}-{w[1]:.2f}s" for w in prior] + [f"{start_time:.2f}-{end_time:.2f}s"]
        # A replayed prefix rewrites remaining_tail(k) in place (PrefixReplayAgentLoop._rewrite_remaining_counts)
        return prompts.TOOL_RESPONSE.format(
            obs_idx=obs_idx,
            start_time=start_time,
            end_time=end_time,
            n_frames=n_kept,
            clip_interval=clip_interval,
            retrieved_windows=", ".join(spans),
            tail=prompts.remaining_tail(k),
        )

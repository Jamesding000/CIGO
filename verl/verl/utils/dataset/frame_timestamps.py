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
"""Timestamp burn-in for the clip frames the RL video tool returns (verl/tools/video_clip_tool.py)."""

from __future__ import annotations

import math
import os

from PIL import ImageDraw, ImageFont


def draw_timestamps(images: list, fps: float, start_time: float = 0.0) -> list:
    """Draw each PIL frame's absolute time (start_time + i / fps seconds) in its bottom-right corner, in place;
    returns the same list."""
    num_images = len(images)
    timestamps = [start_time + i / fps for i in range(num_images)]
    timestamps_str = [f"{t:.2f}s" for t in timestamps]
    fontsize = 40
    sqr_size = math.sqrt(images[0].height * images[0].width)
    fontsize = int(fontsize * sqr_size / 336)
    # Font next to this module (Ray workers' cwd is not the repo root), then repo-relative, then matplotlib's
    fontpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DejaVuSans-Bold.ttf')
    if not os.path.exists(fontpath):
        fontpath = 'verl/verl/utils/dataset/DejaVuSans-Bold.ttf'
    if not os.path.exists(fontpath):
        import matplotlib

        fontpath = os.path.join(
            os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'
        )
    font = ImageFont.truetype(fontpath, fontsize)
    margin = 10
    for image, time_str in zip(images, timestamps_str):
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), time_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = image.width - text_width - margin
        y = image.height - text_height - margin
        draw.text((x, y), time_str, fill=(255, 0, 0), font=font)
    return images

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
"""Qwen3-VL video placeholders in the token-append rollout.

The tool loops keep one raw `<|vision_start|><|video_pad|><|vision_end|>` placeholder per video, which
vLLM expands per frame itself. The HF training forward needs the expanded ids, one <|video_pad|> per
visual feature, so each video's expansion is computed once when the video enters the context and spliced
in at postprocess. The surrounding text is never decoded or re-tokenized, so the splice is byte-exact
outside video regions.
"""

PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def is_qwen3vl(processor) -> bool:
    return processor is not None and processor.__class__.__name__ == "Qwen3VLProcessor"


def video_expansion_ids(processor, video, metadata) -> list[int]:
    """The expanded ids of one video's placeholder; [] for other processors (the splice is then a no-op)."""
    if not is_qwen3vl(processor):
        return []
    out = processor(
        text=[PLACEHOLDER],
        videos=[video],
        video_metadata=[metadata],
        return_tensors="pt",
        do_sample_frames=False,
    )
    return out["input_ids"][0].tolist()


def splice_video_expansions(raw_ids, raw_mask, raw_lp, expansions, vi_start, vision_ids, pad_token_id):
    """Replace each video placeholder in `raw_ids` with its expansion (`expansions[vi_start]` onwards).

    A stray <|video_pad|> outside a placeholder has no visual features and becomes the pad token.
    Returns the spliced ids, mask and log-probs, the next unused video index, and `pre_to_post`
    (the spliced offset of every raw offset, the end included).
    """
    vision_start_id, video_pad_id, vision_end_id = vision_ids
    new_ids: list[int] = []
    new_mask: list[int] = []
    new_lp: list[float] | None = [] if raw_lp is not None else None
    pre_to_post = [0] * (len(raw_ids) + 1)
    i, vi = 0, vi_start
    while i < len(raw_ids):
        pre_to_post[i] = len(new_ids)
        if (
            i + 2 < len(raw_ids)
            and raw_ids[i] == vision_start_id
            and raw_ids[i + 1] == video_pad_id
            and raw_ids[i + 2] == vision_end_id
            and vi < len(expansions)
        ):
            exp = expansions[vi]
            new_ids.extend(exp)
            if raw_mask is not None:
                new_mask.extend([raw_mask[i]] * len(exp))
            if raw_lp is not None:
                new_lp.extend([raw_lp[i] if raw_lp[i] is not None else 0.0] * len(exp))
            pre_to_post[i + 1] = pre_to_post[i + 2] = len(new_ids)
            i += 3
            vi += 1
        else:
            tok = raw_ids[i]
            new_ids.append(pad_token_id if tok == video_pad_id else tok)
            if raw_mask is not None:
                new_mask.append(raw_mask[i])
            if raw_lp is not None:
                new_lp.append(raw_lp[i])
            i += 1
    pre_to_post[len(raw_ids)] = len(new_ids)
    return new_ids, new_mask, new_lp, vi, pre_to_post


def expand_video_placeholders(output, tokenizer, processor, response_length: int) -> None:
    """Splice every video's expansion (extra_fields["per_video_expansion"]) into a Qwen3-VL output's prompt
    and response ids, mask and log-probs, and remap the per-turn spans to the spliced offsets, in place.

    Then check that the response fits `response_length`: tokenizer.pad only pads, and the tool loops cap
    each turn, not the spliced total, so an overlong response fails here rather than at torch.cat.
    """
    # per_video_expansion is present exactly when a tool loop's row holds videos
    if is_qwen3vl(processor) and "per_video_expansion" in output.extra_fields:
        expansions = output.extra_fields["per_video_expansion"]
        video_pad_id = processor.video_token_id
        vision_ids = (
            tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            video_pad_id,
            tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
        pad_id = tokenizer.pad_token_id

        n_prompt_vids = sum(1 for t in output.prompt_ids if t == video_pad_id)
        prompt_ids, _, _, vi, _ = splice_video_expansions(
            output.prompt_ids, None, None, expansions, 0, vision_ids, pad_id
        )
        assert vi == n_prompt_vids, f"prompt splice: consumed {vi} videos but counted {n_prompt_vids}"
        n_raw = len(output.response_ids)
        response_ids, response_mask, response_lp, vi, pre_to_post = splice_video_expansions(
            output.response_ids,
            output.response_mask,
            output.response_logprobs,
            expansions,
            n_prompt_vids,
            vision_ids,
            pad_id,
        )
        n_videos = len(output.multi_modal_data["videos"])
        assert vi == n_videos, f"response splice: consumed {vi} videos total but have {n_videos}"

        for key in ("assistant_turn_spans", "tool_response_spans"):
            if key in output.extra_fields:
                output.extra_fields[key] = [
                    (pre_to_post[s], min(pre_to_post[e], len(response_ids)))
                    for s, e in output.extra_fields[key]
                    if s <= n_raw
                ]

        output.prompt_ids = prompt_ids
        output.response_ids = response_ids
        output.response_mask = response_mask
        if output.response_logprobs is not None:
            output.response_logprobs = response_lp

    assert len(output.response_ids) <= response_length, (
        f"post-splice response_ids length {len(output.response_ids)} exceeds response_length {response_length}"
    )

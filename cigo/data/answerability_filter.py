r"""Stage 2, answerability filter: keep questions whose rollouts disagree, so GRPO groups carry a learning signal.

The model answers each question N times with its overview frames. Multiple choice is kept when some answers are
correct and some are not; grounding when the IoU of its answers spreads by more than --min-iou-spread.

    python -m cigo.data.answerability_filter --model $CIGO_MODEL_DIR/Qwen3-VL-8B-Thinking \
        --input candidates_nonblind.json --frame-dir $CIGO_DATA_ROOT/<dataset>/video_14400frames_fps2 \
        --output candidates_final.json --work-dir work/answerability --num-gpus 4
"""

import argparse

from cigo.data import rollouts


def answerability_stats(row: dict, scores: list) -> dict:
    if rollouts.is_grounding(row):
        return {"iou_spread": max(scores) - min(scores)}
    return {"correct": int(sum(scores)), "n": len(scores)}


def answerable(stats: dict, min_iou_spread: float) -> bool:
    if "iou_spread" in stats:
        return stats["iou_spread"] > min_iou_spread
    return 0 < stats["correct"] < stats["n"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rollouts.add_args(parser, n=8, temperature=1.0)
    parser.add_argument("--frame-dir", required=True, help="frames of the question videos, <dir>/<video id>/")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--min-iou-spread", type=float, default=0.02)
    args = parser.parse_args()
    args.module = "cigo.data.answerability_filter"

    scores = rollouts.sample(args, video_args=(args.frame_dir, args.max_frames, args.max_pixels))
    if args.shard is not None:
        return
    rows = rollouts.load_rows(args.input)
    stats = {r["id"]: answerability_stats(r, scores[r["id"]]) for r in rows if r["id"] in scores}
    rollouts.write(args, rows, stats, keep=lambda s: answerable(s, args.min_iou_spread))


if __name__ == "__main__":
    main()

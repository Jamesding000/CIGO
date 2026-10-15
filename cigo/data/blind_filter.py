r"""Stage 1, blind filter: remove questions the model answers above chance without the video.

The model answers each question N times from its text alone. Multiple choice: k correct of N tested against
Binomial(N, 1/options). Grounding: the mean IoU tested against a guesser that draws start ~ U(0, L) and
end ~ U(start, L). A question is blind, and removed, when the p-value is at most alpha.

    python -m cigo.data.blind_filter --model $CIGO_MODEL_DIR/Qwen3-VL-8B-Thinking \
        --input candidates.json --output candidates_nonblind.json --work-dir work/blind --num-gpus 4
"""

import argparse

import numpy as np
from scipy.stats import binom

from cigo.data import rollouts
from cigo.reward.parse import parse_options


def mcq_p_value(k: int, n: int, n_options: int) -> float:
    """Pr(Binomial(n, 1/n_options) >= k)."""
    return float(binom.sf(k - 1, n, 1.0 / n_options))


def tg_p_value(mean_iou: float, start: float, end: float, duration: float, n: int, draws: int, seed: int) -> float:
    """Monte-Carlo Pr(mean IoU of n random intervals >= mean_iou), as (1 + hits) / (draws + 1)."""
    rng = np.random.default_rng(seed)
    a, b = start / duration, end / duration
    s = rng.uniform(0.0, 1.0, (draws, n))
    e = s + rng.uniform(0.0, 1.0, (draws, n)) * (1.0 - s)
    inter = np.clip(np.minimum(e, b) - np.maximum(s, a), 0.0, None)
    union = np.maximum(e, b) - np.minimum(s, a)
    means = np.where(union > 0, inter / union, 0.0).mean(axis=1)
    return float((1 + (means >= mean_iou).sum()) / (draws + 1))


def blind_stats(row: dict, scores: list, draws: int, seed: int) -> dict:
    """The row's test statistic and p-value; p is None when no chance level is defined."""
    if rollouts.is_grounding(row):
        start, end = row["solution"]
        duration = float(row.get("duration") or 0.0)
        mean_iou = float(np.mean(scores))
        p = tg_p_value(mean_iou, start, end, duration, len(scores), draws, seed) if duration > 0 else None
        return {"mean_iou": mean_iou, "p": p}
    k = int(sum(scores))
    options = parse_options(row["text"])
    p = mcq_p_value(k, len(scores), len(options)) if options else None
    return {"correct": k, "p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rollouts.add_args(parser, n=4, temperature=0.6)
    parser.add_argument("--alpha", type=float, default=0.01, help="significance level of the blind test")
    parser.add_argument("--draws", type=int, default=100000, help="Monte-Carlo draws of the grounding test")
    args = parser.parse_args()
    args.module = "cigo.data.blind_filter"

    scores = rollouts.sample(args)
    if args.shard is not None:
        return
    rows = rollouts.load_rows(args.input)
    stats = {r["id"]: blind_stats(r, scores[r["id"]], args.draws, args.seed) for r in rows if r["id"] in scores}
    # a question without a chance level is kept: it was not shown to be blind
    rollouts.write(args, rows, stats, keep=lambda s: s["p"] is None or s["p"] > args.alpha)


if __name__ == "__main__":
    main()

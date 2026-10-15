"""The main table from the result.csv files: accuracy, mIoU for Charades-STA, and M-Avg (the mean of the seven
per-task accuracies) for MLVU.

    python -m cigo.eval.main_table <output_dir>/<model> [--suffix _stepwise_t3]
"""
import argparse
import os

# benchmark -> (max_pixels, max_frames)
SETTINGS = {
    "videomme": ("224*224", 1024),
    "mlvu": ("224*224", 1024),
    "lvbench": ("224*224", 1024),
    "longvideobench": ("224*224", 1024),
    "videommmu": ("384*384", 256),
    "charades": ("384*384", 256),
}


def headline(benchmark, result_csv):
    """The reported number of one benchmark, from the text of its result.csv."""
    lines = [line.strip() for line in result_csv.splitlines() if line.strip()]
    i = max((k for k, line in enumerate(lines) if line.startswith(("acc", "recall@0.3"))), default=None)
    if i is None:
        return float("nan")
    row = dict(zip(lines[i].split(", "), map(float, lines[i + 1].split(", "))))
    if benchmark == "mlvu":
        tasks = [v for k, v in row.items() if k != "acc"]
        assert len(tasks) == 7, f"MLVU result has {len(tasks)} of its 7 tasks"
        return sum(tasks) / len(tasks)
    return row["miou"] if benchmark == "charades" else row["acc"]


def main_table(model_dir, suffix="_stepwise_t3"):
    """{benchmark: number} for every main-table benchmark that has a result.csv under model_dir."""
    out = {}
    for benchmark, (pixels, frames) in SETTINGS.items():
        path = os.path.join(model_dir, f"evaluation_maxpix{pixels}_maxfrm{frames}_number{suffix}", benchmark,
                            "result.csv")
        if os.path.exists(path):
            with open(path) as f:
                out[benchmark] = headline(benchmark, f.read())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="<output_dir>/<model> of score_vqa / score_tg")
    parser.add_argument("--suffix", default="_stepwise_t3",
                        help="evaluation-name suffix of the setting (the tool-using main table: _stepwise_t3)")
    args = parser.parse_args()
    table = main_table(args.model_dir, args.suffix)
    print("\t".join(SETTINGS))
    print("\t".join(f"{table[b]:.2f}" if b in table else "-" for b in SETTINGS))

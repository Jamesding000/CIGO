#    Copyright 2024 Flash-VStream Authors
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Shared by score_vqa.py and score_tg.py: the command line, one inference process per GPU chunk, reading the
chunk shards back, and writing result.json / result.csv."""

import argparse
import json
import logging
import os
import subprocess


def dataset_info(name, split):
    """Frame directory and ground-truth file of a benchmark, under $CIGO_DATA_ROOT/<name>/."""
    root = os.environ["CIGO_DATA_ROOT"]
    return {"frame_dir": f"{root}/{name}/video_14400frames_fps2", "data_file": f"{root}/{name}/{split}.json"}


def launch_multi_gpu_eval(args, dataset_name, frame_dir, data_file, evaluation_name='evaluation'):
    """Run `args.inference_file` once per GPU chunk on the benchmark; returns its output directory."""
    output_dir = os.path.join(args.output_dir, evaluation_name, dataset_name)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f'launch_multi_gpu_eval: args={args}')
    # chunk k runs on the k-th visible GPU
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(args.num_chunks)]
    if len(gpus) < args.num_chunks:
        raise SystemExit(f"{args.num_chunks} chunks but CUDA_VISIBLE_DEVICES={visible} lists {len(gpus)} GPUs")
    processes = []
    for idx in range(args.num_chunks):
        cmd = ["python3", args.inference_file,
               "--model-path", args.model_path,
               "--video_dir", frame_dir,
               "--gt_file", data_file,
               "--output_dir", output_dir,
               "--output_name", "pred",
               "--num-chunks", str(args.num_chunks),
               "--chunk-idx", str(idx),
               "--max_pixels", str(args.max_pixels),
               ]
        if args.max_frames:
            cmd += ["--max_frames", str(args.max_frames)]
        cmd += ["--prompt_type", args.prompt_type]
        if args.use_tool:
            cmd += ["--use_tool"]
        if args.tool_config:
            cmd += ["--tool_config", args.tool_config]
        logging.debug(f"Starting subprocess with command: {' '.join(cmd)}")
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = gpus[idx]
        # Exit 75 means the chunk stopped early to free vLLM host memory (EVAL_RESTART_EVERY): rerun it.
        p = subprocess.Popen(["bash", "-c", 'while :; do "$@"; rc=$?; [ $rc -eq 75 ] || exit $rc; done',
                              "_", *cmd], env=my_env)
        processes.append(p)
    for idx, p in enumerate(processes):
        stdout, stderr = p.communicate()
        logging.debug(f"Subprocess {idx} stdout: {stdout}")
        if stderr:
            logging.error(f"Subprocess {idx} stderr: {stderr}")
        if p.returncode != 0:
            logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
        else:
            logging.debug(f"Subprocess {idx} completed successfully")
    failed = [idx for idx, p in enumerate(processes) if p.returncode != 0]
    if failed:
        raise SystemExit(f"chunk(s) {failed} of {len(processes)} failed; not scoring a partial benchmark")
    return output_dir


def load_predictions(output_path, gt_file, num_chunks):
    """The prediction rows of every chunk shard whose id is in the ground-truth file."""
    names = [f"{num_chunks}_{idx}.json" for idx in range(num_chunks)] if num_chunks > 1 else ["pred.json"]
    pred_contents = [json.loads(line) for name in names for line in open(os.path.join(output_path, name))]
    gt_id_set = {x['id'] for x in json.load(open(gt_file))}
    kept = [x for x in pred_contents if x['id'] in gt_id_set]
    missing = len(gt_id_set - {x['id'] for x in kept})
    if missing:
        print(f"[main] WARNING {missing} of {len(gt_id_set)} ground-truth items have no prediction")
    return kept


def write_results(output_path, prediction_set, counts, scores):
    """result.json = `prediction_set`; result.csv = "Result:", the `counts` lines, then the mean x 100 of each
    score series in first-seen order, one "series, mean" line each and again as a header row and a value
    row. `scores` is an iterable of (series, value)."""
    json_path = os.path.join(output_path, 'result.json')
    with open(json_path, "w") as f:
        json.dump(prediction_set, f, indent=4)
    print("[main] All evaluation completed!")

    totals = {}
    for key, value in scores:
        total = totals.setdefault(key, [0, 0])
        total[0] += value
        total[1] += 1
    means = {key: f"{score_sum / count * 100:.6f}" for key, (score_sum, count) in totals.items()}
    output = "Result:\n" + "".join(f"{line}\n" for line in counts)
    output += "".join(f"{key}, {mean}\n" for key, mean in means.items()) + "\n"
    output += ", ".join(means) + "\n" + ", ".join(means.values()) + "\n"
    print(output)
    with open(json_path.replace(".json", ".csv"), 'w') as f:
        f.write(output)


def main(get_dataset_info, calc_eval_result):
    """Evaluate `--dataset` (get_dataset_info: name -> {"frame_dir", "data_file"}, None if unknown) and score it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--evaluation_name", type=str, default='evaluation')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--prompt_type", type=str, default="THINK_GENERAL",
                        help="User prompt without the tool, a template name in cigo/prompts.py")
    parser.add_argument("--use_tool", action="store_true",
                        help="Evaluate with the video tool and the prompts it was trained with")
    parser.add_argument("--tool_config", type=str, default=None,
                        help="Tool YAML; must match the one the checkpoint was trained with")
    parser.add_argument("--inference_file", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py"),
                        help="generator launched by path once per GPU chunk")
    args = parser.parse_args()

    info = get_dataset_info(args.dataset)
    if info is None:
        raise SystemExit(f'[main] ERROR {args.dataset} dataset was not found!')
    print(f'[main] Execute {args.dataset} evaluation')
    out_dir = launch_multi_gpu_eval(args, args.dataset, **info, evaluation_name=args.evaluation_name)
    print(f'[main] Execute {args.dataset} evaluation')
    calc_eval_result(out_dir, info['data_file'], args.num_chunks)

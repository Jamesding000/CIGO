#!/usr/bin/env bash
# Main-table evaluation of one model: every benchmark of EVAL_CONFIG, then the table; exits 1 if any benchmark fails.
# A rerun resumes each benchmark from its saved shards, so a results folder holds one model and setting only.
#
#   scripts/eval.sh MODEL [EVAL_CONFIG]
#     MODEL        merged Hugging Face model directory, or a training checkpoint
#                  <run dir>/global_step_<N>[/actor], merged first into $CIGO_EXPORT_DIR
#     EVAL_CONFIG  configs/eval/main_table.yaml (default; the tool-using policies)
#                  or configs/eval/base_notool.yaml (the base model without tools)
#
#   CIGO_DATA_ROOT      benchmark ground truth and frames (<root>/<benchmark>/), required
#   CIGO_EVAL_DIR       results (<dir>/<model tag>/evaluation_*/<benchmark>/), required
#   CIGO_EXPORT_DIR     merged checkpoints (<dir>/<run name>_step<N>), required to merge
#   EVAL_MODEL_TAG      results folder name (default: the model directory's name)
#   BENCHMARKS          space-separated subset of the config's benchmarks
#   NUM_GPUS            inference chunks, one per GPU (default: all visible GPUs)
#   CIGO_SCRATCH        temp root (default: $TMPDIR, else /tmp)
#   EVAL_BATCH_SIZE     inference batch size (default 32 on GPUs with 80 GiB or more, 24 below)
#   EVAL_RESTART_EVERY  batches between vLLM engine restarts (default 2, or 1 on GPUs under 80 GiB)
#   EVAL_NUM_WORKERS    data-loader workers (default 1)
set -uo pipefail
if [ $# -lt 1 ]; then sed -n '2,20p' "$0" >&2; exit 2; fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# resolved before the cd; -s keeps symlinks, so the model tag is the name given
MODEL=$(realpath -ms "$1")
EVAL_CONFIG=$(realpath -ms "${2:-$REPO/configs/eval/main_table.yaml}")
: "${CIGO_DATA_ROOT:?}" "${CIGO_EVAL_DIR:?}"
export CIGO_DATA_ROOT
cd "$REPO"
# this checkout's verl and cigo, whatever the environment has installed; the per-GPU workers inherit it
export PYTHONPATH=$REPO/verl:$REPO

# --- the model: merge a training checkpoint first ---------------------------------------------
if [ ! -e "$MODEL/config.json" ] || [ ! -e "$MODEL/tokenizer_config.json" ]; then
    [ "$(basename "$MODEL")" = actor ] && MODEL=$(dirname "$MODEL")
    step=$(basename "$MODEL")
    [[ $step == global_step_* && -d $MODEL/actor ]] || { echo "$MODEL is neither a model nor a global_step_<N> checkpoint" >&2; exit 2; }
    : "${CIGO_EXPORT_DIR:?needed to merge $MODEL}"
    export_dir=$CIGO_EXPORT_DIR/$(basename "$(dirname "$MODEL")")_step${step#global_step_}
    # the merger writes tokenizer_config.json last, so an interrupted merge is redone
    if [ ! -e "$export_dir/tokenizer_config.json" ]; then
        python -m verl.model_merger merge --backend fsdp \
            --local_dir "$MODEL/actor" --target_dir "$export_dir" || exit $?
    fi
    MODEL=$export_dir
fi
tag=${EVAL_MODEL_TAG:-$(basename "$MODEL")}

# --- the setting -------------------------------------------------------------------------------
setting=$(python - "$EVAL_CONFIG" <<'PY'
import shlex, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(f"use_tool={'1' if c['use_tool'] else ''}")
print(f"prompt_type={shlex.quote(c.get('prompt_type') or '')}")
print(f"tool_config={shlex.quote(c.get('tool_config') or '')}")
print(f"suffix={shlex.quote(c['suffix'])}")
for k, v in c["env"].items():
    print(f"export {k}={shlex.quote(str(v))}")
print("settings=(" + " ".join(shlex.quote(f"{b}:{s['max_pixels']}:{s['max_frames']}") for b, s in c["benchmarks"].items()) + ")")
PY
) || { echo "cannot read $EVAL_CONFIG" >&2; exit 2; }
eval "$setting"
extra=()
[ -n "$use_tool" ] && extra+=(--use_tool)
[ -n "$prompt_type" ] && extra+=(--prompt_type "$prompt_type")
[ -n "$tool_config" ] && extra+=(--tool_config "$tool_config")

# --- runtime -----------------------------------------------------------------------------------
SCRATCH=${CIGO_SCRATCH:-${TMPDIR:-/tmp}}
export TMPDIR=$SCRATCH/cigo_eval_$$
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
# GPUs: NUM_GPUS, else CUDA_VISIBLE_DEVICES, else all
IFS=, read -ra visible <<< "${CUDA_VISIBLE_DEVICES:-}"
n_gpus=${NUM_GPUS:-${visible[0]:+${#visible[@]}}}
n_gpus=${n_gpus:-$(nvidia-smi -L | wc -l)}
# vLLM host memory grows until the engine restarts, so a chunk exits every EVAL_RESTART_EVERY batches and resumes
card_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
if [ "${card_mib:-0}" -ge 81920 ]; then
    export EVAL_RESTART_EVERY=${EVAL_RESTART_EVERY:-2}
else
    export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-24} EVAL_RESTART_EVERY=${EVAL_RESTART_EVERY:-1}
fi
export EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-1}
echo "eval: model=$MODEL tag=$tag config=$EVAL_CONFIG chunks=$n_gpus card=${card_mib}MiB" \
     "batch=${EVAL_BATCH_SIZE:-default} restart=$EVAL_RESTART_EVERY workers=$EVAL_NUM_WORKERS"

# --- the benchmarks ----------------------------------------------------------------------------
failures=""
for s in "${settings[@]}"; do
    IFS=: read -r bench pixels frames <<< "$s"
    if [ -n "${BENCHMARKS:-}" ] && [[ " $BENCHMARKS " != *" $bench "* ]]; then continue; fi
    scorer=cigo/eval/score_vqa.py
    [ "$bench" = charades ] && scorer=cigo/eval/score_tg.py
    echo "==== $bench | $pixels px | $frames frames | ${use_tool:+tool}${prompt_type}"
    python "$scorer" \
        --inference_file cigo/eval/inference.py \
        --model-path "$MODEL" \
        --dataset "$bench" \
        --output_dir "$CIGO_EVAL_DIR/$tag" \
        --evaluation_name "evaluation_maxpix${pixels}_maxfrm${frames}_number${suffix}" \
        --num_chunks "$n_gpus" \
        --max_pixels $(( pixels )) \
        --max_frames "$frames" "${extra[@]}"
    rc=$?
    # a failed benchmark does not stop the others; the exit status reports it
    [ $rc -eq 0 ] || { failures="$failures $bench(rc=$rc)"; echo "!!!! FAILED: $bench rc=$rc"; }
done

python -m cigo.eval.main_table "$CIGO_EVAL_DIR/$tag" --suffix "$suffix"
if [ -n "$failures" ]; then
    echo "EVAL FAILURES:$failures" >&2
    exit 1
fi

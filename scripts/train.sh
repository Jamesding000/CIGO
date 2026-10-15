#!/usr/bin/env bash
# Train one arm (configs/, 4 GPUs) with verl.trainer.main_ppo, then merge the last checkpoint; exit status is the trainer's.
#
#   scripts/train.sh ARM EXP_NAME [hydra overrides ...]
#     ARM       grpo | grpo_ar | grpo_ir | ct | cigo   (configs/arm/)
#     EXP_NAME  run name; a run directory that already holds a checkpoint is resumed
#
#   CIGO_DATA_ROOT      training frames (<root>/<corpus>/video_14400frames_fps2), required
#   CIGO_MODEL_DIR      base model (<dir>/Qwen3-VL-8B-Thinking), required
#   CIGO_OUTPUT_DIR     run directories (<dir>/EXP_NAME: checkpoints, rollouts), required
#   CIGO_EXPORT_DIR     merged final model (<dir>/EXP_NAME), required
#   CIGO_SCRATCH        root of the job's temp dirs (default: $TMPDIR, else /tmp)
#   JOB_ID              temp-dir suffix (default: the scheduler's job id, else local<pid>)
#   LOGGER_OUTPUT_FILE  reward log
#   WANDB_API_KEY       for the wandb logger
set -uo pipefail
if [ $# -lt 2 ]; then sed -n '2,15p' "$0" >&2; exit 2; fi
ARM=$1 EXP_NAME=$2
shift 2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${CIGO_DATA_ROOT:?}" "${CIGO_MODEL_DIR:?}" "${CIGO_OUTPUT_DIR:?}" "${CIGO_EXPORT_DIR:?}"
[ -f "$REPO/configs/arm/$ARM.yaml" ] || { echo "no configs/arm/$ARM.yaml" >&2; exit 2; }
export CIGO_REPO=$REPO CIGO_DATA_ROOT CIGO_MODEL_DIR CIGO_OUTPUT_DIR
# configs/base.yaml names the data files and dataset / sampler classes relative to the repository root
cd "$REPO"
# this checkout's verl and cigo, inherited by the Ray workers
export PYTHONPATH=$REPO/verl:$REPO
# refuse to train on code imported from elsewhere (e.g. an editable install)
python -c "import sys, verl, cigo; bad = [m.__file__ for m in (verl, cigo) if not m.__file__.startswith(sys.argv[1] + '/')]; print('code from', sys.argv[1]) if not bad else sys.exit('imported from outside the repository: ' + ', '.join(bad))" "$REPO" \
    || exit 3
SAVE_PATH=$CIGO_OUTPUT_DIR/$EXP_NAME
JOB=${JOB_ID:-${SLURM_JOB_ID:-local$$}}
SCRATCH=${CIGO_SCRATCH:-${TMPDIR:-/tmp}}

# --- job-scoped temp dirs ---------------------------------------------------------------------
# per-job paths, so two jobs on one node share no IPC socket or Ray directory
export TMPDIR=$SCRATCH/cigo_job_$JOB
mkdir -p "$TMPDIR"
export VERL_ZMQ_IPC_DIR=${VERL_ZMQ_IPC_DIR:-$TMPDIR}
# short: Ray appends ~65 bytes and an AF_UNIX socket path is capped at 107
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rj_${USER:-$(id -un)}_$JOB}
mkdir -p "$RAY_TMPDIR" 2>/dev/null || { export RAY_TMPDIR=$SCRATCH/r$JOB; mkdir -p "$RAY_TMPDIR"; }
# the reward log, in the first writable directory
if [ -z "${LOGGER_OUTPUT_FILE:-}" ]; then
    for d in /tmp/verl_reward_logs "$SCRATCH/verl_reward_logs" "$HOME/.cache/verl_reward_logs"; do
        mkdir -p "$d" 2>/dev/null && export LOGGER_OUTPUT_FILE=$d/rewards.log && break
    done
fi

# --- runtime ----------------------------------------------------------------------------------
export VLLM_USE_V1=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_MM_INPUT_CACHE_GIB=0
# vLLM's sleep mode (CuMemAllocator) fails with expandable_segments
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 VERL_LOGGING_LEVEL=INFO RAY_DEDUP_LOGS=0
# one wandb run per EXP_NAME, continued across restarts
export WANDB_RUN_ID=${WANDB_RUN_ID:-$(printf '%s\n' "$EXP_NAME" | md5sum | cut -c1-16)} WANDB_RESUME=allow
export WANDB_DIR=$SAVE_PATH WANDB_CONFIG_DIR=$SAVE_PATH
mkdir -p "$SAVE_PATH"
echo "train: arm=$ARM run=$SAVE_PATH tmp=$TMPDIR ray=$RAY_TMPDIR log=${LOGGER_OUTPUT_FILE:-}"

python -m verl.trainer.main_ppo --config-path="$REPO/configs" --config-name=base \
    arm="$ARM" trainer.experiment_name="$EXP_NAME" "$@"
rc=$?
if [ $rc -ne 0 ]; then
    echo "TRAINING FAILED (exit $rc): no merge. Last checkpoint:" \
         "$(cat "$SAVE_PATH/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)" >&2
    exit $rc
fi

# --- merge the last checkpoint into a Hugging Face model --------------------------------------
step=$(ls -d "$SAVE_PATH"/global_step_*/ 2>/dev/null | xargs -rn1 basename | sort -V | tail -1)
if [ -z "$step" ]; then
    echo "no checkpoint in $SAVE_PATH (save_freq off?): nothing to merge"
    exit 0
fi
target=$CIGO_EXPORT_DIR/$EXP_NAME
# re-merge when the checkpoint is newer than the export; tokenizer_config.json is written last
if [ ! -e "$target/tokenizer_config.json" ] || [ "$SAVE_PATH/$step" -nt "$target/tokenizer_config.json" ]; then
    python -m verl.model_merger merge --backend fsdp --local_dir "$SAVE_PATH/$step/actor" \
        --target_dir "$target" || exit $?
fi
echo "merged $SAVE_PATH/$step -> $target"

#!/bin/bash
# Generate only 60 new clips; retain 40 existing clips in place.
# Submit from the wanthesis repository root. This job never submits another job.
#SBATCH --job-name=walking_pov_mixed_100
#SBATCH --account=gt69
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dhiazr/walking_pov_mixed_100_%j.log
#SBATCH --open-mode=append

set -euo pipefail

STAGE=all
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    STAGE=$1
    shift
fi
WAN_VENV=${WAN_VENV:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/envs/wan22}
WAN_CACHE=${WAN_CACHE:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/hf_cache}

case "$STAGE" in
    all|check|generate|label|status) ;;
    *) echo "ERROR: choose all, check, generate, label or status." >&2; exit 2 ;;
esac
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        echo "Submit from the wanthesis repository root:"
        echo "  sbatch jobs/datasets/create_walking_pov_mixed_100.sh"
        echo "  sbatch jobs/datasets/create_walking_pov_mixed_100.sh --limit 4"
        echo "  bash jobs/datasets/create_walking_pov_mixed_100.sh check"
        echo "  bash jobs/datasets/create_walking_pov_mixed_100.sh status"
        echo "Stages: all (default), check, generate, label, status."
        echo "Options: --out-root PATH, --source-root PATH, --prompts PATH, --model-dir PATH, --yolo-model PATH, --limit N."
        echo "Use WAN_VENV and WAN_CACHE to select an existing environment and cache."
        exit 0
    fi
done
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    echo "ERROR: use one GPU and no job array." >&2
    exit 1
fi

# Slurm copies the wrapper; fall back to the directory where sbatch was run.
WRAPPER_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_DIRECTORY=$(cd -- "$WRAPPER_DIRECTORY/../.." && pwd)
SCRIPT_RELATIVE_PATH=src/datasets/create_walking_pov_mixed_100.py
if [[ ! -f "$REPOSITORY_DIRECTORY/$SCRIPT_RELATIVE_PATH" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPOSITORY_DIRECTORY=$SLURM_SUBMIT_DIR
fi
SCRIPT=$REPOSITORY_DIRECTORY/$SCRIPT_RELATIVE_PATH
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: run or submit from the wanthesis repository root." >&2
    exit 1
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$REPOSITORY_DIRECTORY/src
cd -- "$REPOSITORY_DIRECTORY"

# These stages use only the Python standard library and can run on a login node.
if [[ "$STAGE" == check || "$STAGE" == status ]]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 9))'; then
        exec python3 -B "$SCRIPT" "$STAGE" "$@"
    fi
    if ! type module >/dev/null 2>&1; then
        echo "ERROR: Python 3.9 or newer is needed. Load miniforge3, then retry." >&2
        exit 1
    fi
    module load miniforge3
    exec python -B "$SCRIPT" "$STAGE" "$@"
fi

if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: submit all/generate/label with sbatch on one GPU." >&2
    exit 1
fi
if [[ "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "ERROR: this wrapper requires one task." >&2
    exit 1
fi
if [[ "$WAN_VENV" != /* || ! -x "$WAN_VENV/bin/python" || ! -f "$WAN_VENV/bin/activate" ]]; then
    echo "ERROR: the existing Wan environment is missing: $WAN_VENV" >&2
    exit 1
fi

module purge
module load pytorch/2.7.0
source "$WAN_VENV/bin/activate"
export HF_HOME=$WAN_CACHE
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
DATASET_TEMP_ROOT=${SLURM_TMPDIR:-/tmp}/walking_pov_mixed_100_$SLURM_JOB_ID
export TMPDIR=$DATASET_TEMP_ROOT/tmp
export XDG_CACHE_HOME=$DATASET_TEMP_ROOT/cache
export YOLO_CONFIG_DIR=$DATASET_TEMP_ROOT/yolo
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$YOLO_CONFIG_DIR"

python -B - <<'PY'
import torch
import diffusers
import transformers
import imageio_ffmpeg
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise SystemExit("ERROR: no CUDA GPU is available in this Slurm allocation.")
if torch.cuda.device_count() != 1:
    raise SystemExit("ERROR: this job must have exactly one visible GPU.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
PY

# This is useful quota information, but availability of user_info is not required.
if command -v user_info >/dev/null 2>&1; then
    user_info || echo "user_info did not finish successfully; check project quota separately."
fi
echo "Filesystem free space is not the same as shared project quota."

if [[ "$STAGE" == all ]]; then
    python -B -u "$SCRIPT" check "$@"
    python -B -u "$SCRIPT" generate "$@"
    # The Wan process has exited before YOLO starts, releasing its GPU memory.
    python -B -u "$SCRIPT" label "$@"
    python -B -u "$SCRIPT" status "$@"
else
    python -B -u "$SCRIPT" "$STAGE" "$@"
fi

echo "Dataset stage completed. Review the videos and masks before training."
echo "No additional job is submitted automatically."

#!/bin/bash
# One GPU, no automatic submissions. Default: prepare, train, then examples.
#SBATCH --job-name=joint_semantic
#SBATCH --account=gt69
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dhiazr/joint_semantic_%j.log
#SBATCH --open-mode=append
set -euo pipefail

STAGE=all
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    STAGE=$1
    shift
fi
case "$STAGE" in all|prepare|train|extend|examples|check) ;; *) echo 'Choose all, prepare, train, extend, examples or check.' >&2; exit 2 ;; esac
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        echo 'From wanthesis: sbatch jobs/joint_semantic_mixed/run.sh'
        echo 'Stages: all (default), prepare, train, examples; check is read-only.'
        echo 'Resume after a job ends: sbatch jobs/joint_semantic_mixed/run.sh --resume'
        echo 'Continue the completed 30-epoch run: sbatch jobs/joint_semantic_mixed/continue_30epochs.sh'
        echo 'Options: --codec-run PATH --weights-dir PATH --outputs-dir PATH --model-dir PATH --yolo-model PATH'
        echo '         --updates 2400 --evaluate-every 200 --preview-every 400 --learning-rate 0.0002 --new-prompt TEXT'
        exit 0
    fi
done
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo 'Use one task with no job array.' >&2
    exit 1
fi
if [[ "$STAGE" != check && ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo 'Submit prepare/train/examples through sbatch on one GPU.' >&2
    exit 1
fi
WRAPPER_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_DIRECTORY=$(cd -- "$WRAPPER_DIRECTORY/../.." && pwd)
if [[ ! -f "$REPOSITORY_DIRECTORY/src/joint_semantic_mixed/run.py" ]]; then
    REPOSITORY_DIRECTORY=${SLURM_SUBMIT_DIR:?Submit from the wanthesis repository root}
fi
if [[ ! -f "$REPOSITORY_DIRECTORY/src/joint_semantic_mixed/run.py" ]]; then
    echo 'Submit from the wanthesis repository root.' >&2
    exit 1
fi
module purge
module load pytorch/2.7.0
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source "${WAN_VENV:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/envs/wan22}/bin/activate"
export PYTHONPATH=$REPOSITORY_DIRECTORY/src
export HF_HOME=${WAN_CACHE:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/hf_cache}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/joint_semantic_plot_${SLURM_JOB_ID:-check}
export YOLO_CONFIG_DIR=${SLURM_TMPDIR:-/tmp}/joint_semantic_yolo_${SLURM_JOB_ID:-check}
export XDG_CACHE_HOME=${SLURM_TMPDIR:-/tmp}/joint_semantic_cache_${SLURM_JOB_ID:-check}
export YOLO_AUTOINSTALL=false
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$XDG_CACHE_HOME"
cd -- "$REPOSITORY_DIRECTORY"
python -B -c 'import torch, numpy, PIL, matplotlib, imageio, imageio_ffmpeg; print("Joint semantic dependencies available.")'
if [[ "$STAGE" == all || "$STAGE" == train || "$STAGE" == extend || "$STAGE" == examples ]]; then
    python -B -c 'import cv2, ultralytics; print("YOLO evaluation dependencies available.")'
fi
if [[ "$STAGE" == all ]]; then
    python -B -u src/joint_semantic_mixed/run.py prepare "$@"
    # Fresh processes release codec/Wan GPU memory between stages.
    python -B -u src/joint_semantic_mixed/run.py train "$@"
    python -B -u src/joint_semantic_mixed/run.py examples "$@"
elif [[ "$STAGE" == extend ]]; then
    python -B -u src/joint_semantic_mixed/run.py extend "$@"
    python -B -u src/joint_semantic_mixed/run.py examples "$@"
else
    python -B -u src/joint_semantic_mixed/run.py "$STAGE" "$@"
fi

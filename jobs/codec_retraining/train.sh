#!/bin/bash
# Submit from the wanthesis repository root after completing the prepared review.
#SBATCH --job-name=codec_large9
#SBATCH --account=gt69
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dhiazr/codec_large9_%j.log
#SBATCH --open-mode=append

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_JOB_ID:-}" || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "Submit one GPU training job with sbatch; no array."
    exit 1
fi
WRAPPER_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_DIRECTORY=$(cd -- "$WRAPPER_DIRECTORY/../.." && pwd)
if [[ ! -f "$REPOSITORY_DIRECTORY/src/codec_retraining/run.py" ]]; then
    REPOSITORY_DIRECTORY=${SLURM_SUBMIT_DIR:?Submit from the wanthesis repository root}
fi
if [[ ! -f "$REPOSITORY_DIRECTORY/src/codec_retraining/run.py" ]]; then
    echo "Submit from the wanthesis repository root."
    exit 1
fi
module purge
module load pytorch/2.7.0
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${WAN_VENV:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/envs/wan22}/bin/activate"
export PYTHONPATH=$REPOSITORY_DIRECTORY/src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/codec_plot_$SLURM_JOB_ID
cd -- "$REPOSITORY_DIRECTORY"
# Check this before training, so missing plotting dependencies fail early.
python -B -c 'import torch, numpy, PIL, matplotlib; print("Codec dependencies available.")'
python -B -u src/codec_retraining/run.py train "$@"

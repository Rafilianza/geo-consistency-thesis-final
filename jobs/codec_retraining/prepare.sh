#!/bin/bash
# Submit from the wanthesis repository root after dataset generation/export ends.
#SBATCH --job-name=codec_prepare
#SBATCH --account=gt69
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dhiazr/codec_prepare_%j.log
#SBATCH --open-mode=append

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_JOB_ID:-}" || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo "Submit one CPU preparation job with sbatch; no array."
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
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
cd -- "$REPOSITORY_DIRECTORY"
python -B -u src/codec_retraining/run.py prepare "$@"

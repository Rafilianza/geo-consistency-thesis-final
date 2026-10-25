#!/bin/bash
# Continue the completed 30-epoch CNN run to 60 epochs, in separate folders.
#SBATCH --job-name=joint_semantic_extend
#SBATCH --account=gt69
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dhiazr/joint_semantic_extend_%j.log
#SBATCH --open-mode=append
set -euo pipefail

WRAPPER_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_DIRECTORY=$(cd -- "$WRAPPER_DIRECTORY/../.." && pwd)
if [[ ! -f "$REPOSITORY_DIRECTORY/jobs/joint_semantic_mixed/run.sh" ]]; then
    REPOSITORY_DIRECTORY=${SLURM_SUBMIT_DIR:?Submit from the wanthesis repository root}
fi
exec bash "$REPOSITORY_DIRECTORY/jobs/joint_semantic_mixed/run.sh" extend \
    --continue-from /home/dhiazr/joint_semantic_mixed_v1 \
    --weights-dir /home/dhiazr/joint_semantic_mixed_v1_epoch60 \
    --outputs-dir /home/dhiazr/gt69_scratch2/dhiazgeo/joint_semantic_mixed_v1_epoch60 \
    --updates 4800 --resume "$@"

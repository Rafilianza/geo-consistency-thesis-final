#!/bin/bash
# One GPU executes successive arms from the shared FIFO. Use submit.sh.
#SBATCH --job-name=rgbgeomqual_worker
#SBATCH --account=gt69
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

repository=${WAN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ "$repository" != /* || ! -f "$repository/jobs/rgb_geometry_qualitative/common.sh" ]]; then
    echo 'Submit from the wanthesis root, or set WAN_REPO_ROOT to that checkout.' >&2
    exit 1
fi
worker=${1:?The submission helper supplies worker_1 through worker_4.}
shift
case "$worker" in worker_1|worker_2|worker_3|worker_4) ;; *) exit 2 ;; esac
source "$repository/jobs/rgb_geometry_qualitative/common.sh" "$worker" "$@"

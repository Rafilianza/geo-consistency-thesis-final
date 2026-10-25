#!/bin/bash
# Manual fresh-root recovery or video generation. This never submits other jobs.
#SBATCH --job-name=rgbgeomqual_single
#SBATCH --account=gt69
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

repository=${WAN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ "$repository" != /* || ! -f "$repository/jobs/rgb_geometry_qualitative/common.sh" ]]; then
    echo 'Submit from the wanthesis root, or set WAN_REPO_ROOT to that checkout.' >&2
    exit 1
fi
experiment=${1:?Provide baseline, rloo_R, rloo_S, flash_grpo_R, flash_grpo_S, flow_grpo_R or flow_grpo_S.}
shift
case "$experiment" in
    baseline|rloo_R|rloo_S|flash_grpo_R|flash_grpo_S|flow_grpo_R|flow_grpo_S) ;;
    *) echo 'Unknown single-arm experiment.' >&2; exit 2 ;;
esac
export RGB_GEOMETRY_QUALITATIVE_SINGLE_ARM=1
source "$repository/jobs/rgb_geometry_qualitative/common.sh" "$experiment" "$@"

#!/bin/bash
# One fixed LoRA arm; submitted by the separate second-stage launcher.
#SBATCH --job-name=rgbgeomqualora_arm
#SBATCH --account=gt69
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

repository=${WAN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ "$repository" != /* || ! -f "$repository/jobs/rgb_geometry_qualitative_lora/common.sh" ]]; then
    echo 'Submit from the wanthesis root, or set WAN_REPO_ROOT to that checkout.' >&2
    exit 1
fi
experiment=${1:?Provide rloo_L, flash_grpo_L or flow_grpo_L.}
shift
case "$experiment" in
    rloo_L|flash_grpo_L|flow_grpo_L) ;;
    *) echo 'Unknown single-arm experiment.' >&2; exit 2 ;;
esac
source "$repository/jobs/rgb_geometry_qualitative_lora/common.sh" "$experiment" "$@"

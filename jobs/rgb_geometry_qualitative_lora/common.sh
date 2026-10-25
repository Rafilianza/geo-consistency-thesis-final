#!/bin/bash
# Fixed-stage dispatch. Real work is logged before any module/environment setup.
set -euo pipefail
experiment=${1:?Use an experiment-specific wrapper.}
shift
case "$experiment" in
    report|rloo_L|flash_grpo_L|flow_grpo_L) ;;
    *) echo 'Unknown fixed campaign stage.' >&2; exit 2 ;;
esac
arguments=("$@")
read_only=0
output_root=''
reference_root=''
while (( $# > 0 )); do
    case "$1" in
        --plan|--check-inputs|--check-output|--help|-h) read_only=1; shift ;;
        --config|--output-root|--reference-root)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { echo "Provide a value after $1." >&2; exit 2; }
            if [[ "$1" == --output-root ]]; then output_root=$2; fi
            if [[ "$1" == --reference-root ]]; then reference_root=$2; fi
            shift 2 ;;
        --config=*|--output-root=*|--reference-root=*)
            [[ -n "${1#*=}" ]] || { echo "Provide a value in $1." >&2; exit 2; }
            if [[ "$1" == --output-root=* ]]; then output_root=${1#*=}; fi
            if [[ "$1" == --reference-root=* ]]; then reference_root=${1#*=}; fi
            shift ;;
        --resume-from|--checkpoint)
            [[ "${RGB_GEOMETRY_QUALITATIVE_LORA_SINGLE_ARM:-0}" == 1 ]] || { echo 'Checkpoint overrides require single_arm.sh.' >&2; exit 2; }
            (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { echo "Provide a value after $1." >&2; exit 2; }
            shift 2 ;;
        --resume-from=*|--checkpoint=*)
            [[ "${RGB_GEOMETRY_QUALITATIVE_LORA_SINGLE_ARM:-0}" == 1 ]] || { echo 'Checkpoint overrides require single_arm.sh.' >&2; exit 2; }
            [[ -n "${1#*=}" ]] || { echo "Provide a value in $1." >&2; exit 2; }
            shift ;;
        *) echo "Unsupported override: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo 'Use one task with no job array.' >&2
    exit 1
fi
if [[ "$repository" != /* || ! -f "$repository/src/rgb_geometry_qualitative_lora/run.py" ]]; then
    echo 'Submit from the wanthesis root, or set WAN_REPO_ROOT to that checkout.' >&2
    exit 1
fi
cd -P -- "$repository"
repository=$PWD
export WAN_REPO_ROOT=$repository
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$repository/src
if [[ "$read_only" == 1 ]]; then
    exec python3 -B -u -m rgb_geometry_qualitative_lora.run --experiment "$experiment" "${arguments[@]}"
fi
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo 'Submit with sbatch; do not run campaign work on a login node.' >&2
    exit 1
fi
if [[ "$output_root" != /* || "$reference_root" != /* ]]; then
    echo 'Provide absolute --output-root and --reference-root paths.' >&2
    exit 2
fi
# A rejected protected path must not receive even the launcher's CSV log.
python3 -B -u -m rgb_geometry_qualitative_lora.run --experiment "$experiment" \
    --check-output "${arguments[@]}"
if [[ "${RGB_GEOMETRY_QUALITATIVE_LORA_SINGLE_ARM:-0}" == 1 ]]; then
    # Manual recovery/inference starts a new root and never alters the source run.
    python3 -B - "$output_root" <<'PY'
from pathlib import Path
import shutil
import sys
root = Path(sys.argv[1])
if root.exists() or root.is_symlink() or not root.parent.is_dir():
    raise SystemExit('Use a fresh output root whose parent already exists.')
if shutil.disk_usage(root.parent).free < 5_000_000_000:
    raise SystemExit('At least 5 GB of free filesystem space is required; user quota is not checked.')
PY
fi
python3 -B -u -m rgb_geometry_qualitative_lora.run --experiment "$experiment" \
    --check-inputs "${arguments[@]}"
log_directory=$output_root/$experiment
exec python3 -B -u -m rgb_geometry_campaign.launcher \
    --output-dir "$log_directory" -- \
    bash "$repository/jobs/rgb_geometry_qualitative_lora/environment.sh" "$experiment" "${arguments[@]}"

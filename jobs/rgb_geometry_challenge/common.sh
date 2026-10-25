#!/bin/bash
# Fixed-stage dispatch; the CSV launcher records environment setup failures.
set -euo pipefail
experiment=${1:?Use worker.sh or report.sh.}
shift
case "$experiment" in
    worker_1|worker_2|worker_3|worker_4|report) ;;
    *) echo 'Unknown challenge stage.' >&2; exit 2 ;;
esac
arguments=("$@")
output_root=''
reference_root=''
lora_root=''
while (( $# > 0 )); do
    case "$1" in
        --config|--output-root|--reference-root|--lora-root)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || { echo "Provide a value after $1." >&2; exit 2; }
            if [[ "$1" == --output-root ]]; then output_root=$2; fi
            if [[ "$1" == --reference-root ]]; then reference_root=$2; fi
            if [[ "$1" == --lora-root ]]; then lora_root=$2; fi
            shift 2 ;;
        --config=*|--output-root=*|--reference-root=*|--lora-root=*)
            [[ -n "${1#*=}" ]] || { echo "Provide a value in $1." >&2; exit 2; }
            if [[ "$1" == --output-root=* ]]; then output_root=${1#*=}; fi
            if [[ "$1" == --reference-root=* ]]; then reference_root=${1#*=}; fi
            if [[ "$1" == --lora-root=* ]]; then lora_root=${1#*=}; fi
            shift ;;
        *) echo "Unsupported override: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || "${SLURM_NTASKS:-1}" != 1 ]]; then
    echo 'Use one task with no job array.' >&2
    exit 1
fi
if [[ "$repository" != /* || ! -f "$repository/src/rgb_geometry_challenge/run.py" ]]; then
    echo 'Submit from the wanthesis root, or set WAN_REPO_ROOT to that checkout.' >&2
    exit 1
fi
cd -P -- "$repository"
repository=$PWD
export WAN_REPO_ROOT=$repository
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$repository/src
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo 'Submit with submit.sh; do not run inference on a login node.' >&2
    exit 1
fi
if [[ "$output_root" != /* || "$reference_root" != /* || "$lora_root" != /* ]]; then
    echo 'Provide absolute output, reference and LoRA roots.' >&2
    exit 2
fi
python3 -B -u -m rgb_geometry_challenge.run --experiment baseline --check-output "${arguments[@]}"
log_directory=$output_root/report
if [[ "$experiment" == worker_* ]]; then log_directory=$output_root/workers/$experiment; fi
exec python3 -B -u -m rgb_geometry_campaign.launcher \
    --output-dir "$log_directory" -- \
    bash "$repository/jobs/rgb_geometry_challenge/environment.sh" "$experiment" "${arguments[@]}"

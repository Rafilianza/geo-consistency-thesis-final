#!/bin/bash
# This script runs inside the CSV logger, including setup failures.
set -euo pipefail
experiment=${1:?Missing experiment.}
shift
repository=${WAN_REPO_ROOT:?Missing repository.}
cd -P -- "$repository"
repository=$PWD
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$repository/src
case "$experiment" in
    report)
        python3 -B "$repository/jobs/rgb_geometry_qualitative/worker.py" --finalize "$@"
        exec python3 -B -u -m rgb_geometry_qualitative.run --experiment "$experiment" "$@" ;;
esac
if [[ "${SLURM_GPUS_ON_NODE:-1}" != 1 || "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
    echo 'Each experiment requires exactly one GPU on one node.' >&2
    exit 1
fi
wan_environment=${WAN_VENV:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/envs/wan22}
if [[ "$wan_environment" != /* || ! -f "$wan_environment/bin/activate" || ! -x "$wan_environment/bin/python" ]]; then
    echo "Existing Wan environment is missing: $wan_environment" >&2
    exit 1
fi
module purge
module load pytorch/2.7.0
source "$wan_environment/bin/activate"
unset PYTHONHOME
export PYTHONPATH=$repository/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${WAN_CACHE:-/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/hf_cache}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export YOLO_AUTOINSTALL=false MPLBACKEND=Agg
temporary_root=${SLURM_TMPDIR:-/tmp}/rgb_geometry_qualitative_${experiment}_${SLURM_JOB_ID}
export TMPDIR=$temporary_root
export MPLCONFIGDIR=$temporary_root/matplotlib
export YOLO_CONFIG_DIR=$temporary_root/yolo
export XDG_CACHE_HOME=$temporary_root/cache
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$XDG_CACHE_HOME"
if [[ "$experiment" == worker_* ]]; then
    exec "$wan_environment/bin/python" -B -u "$repository/jobs/rgb_geometry_qualitative/worker.py" \
        --worker "$experiment" "$@"
fi
exec "$wan_environment/bin/python" -B -u -m rgb_geometry_qualitative.run --experiment "$experiment" "$@"

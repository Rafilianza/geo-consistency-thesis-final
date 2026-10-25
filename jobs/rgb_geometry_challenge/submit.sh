#!/bin/bash
# Run this helper yourself after syncing the checkout to M3.
set -euo pipefail
repository=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
export WAN_REPO_ROOT=$repository
cd -- "$repository"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$repository/jobs/rgb_geometry_challenge/submit.py" "$@"

#!/bin/bash

#SBATCH --job-name=static-hubs-group
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/output/%A_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/errors/%A_%x_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Aggregate subject-level collapsed network FC outputs and make a group circular plot.
# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Hubness/Static/static_hubs_group_SLURM.sh

# Environment variable options:
#   OUTPUT_DIR=/ptmp/.../outputs/hubness
#   EDGE_THRESHOLD_PCT=50
#   HUB_SELECTION_METRIC=strength   # or participation
#   SUBJECTS="01 02 03"           # optional, defaults to auto-discovery from sub-*/static

set -euo pipefail

SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/Static/static_hubs_group.py"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
EDGE_THRESHOLD_PCT="${EDGE_THRESHOLD_PCT:-50}"
HUB_SELECTION_METRIC="${HUB_SELECTION_METRIC:-strength}"
SUBJECTS="${SUBJECTS:-}"

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/errors

echo "=========================================="
echo "Static Hubs Group Aggregation"
echo "=========================================="
echo "Script: ${SCRIPT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Edge Threshold Percentile: ${EDGE_THRESHOLD_PCT}"
echo "Hub Selection Metric: ${HUB_SELECTION_METRIC}"
if [[ -n "${SUBJECTS}" ]]; then
    echo "Subjects: ${SUBJECTS}"
else
    echo "Subjects: auto-discovery from subject collapsed FC files"
fi
echo "Container: ${CONTAINER}"
echo "=========================================="

CMD=(
    "python" "$SCRIPT"
    --output-dir "$OUTPUT_DIR"
    --edge-threshold-percentile "$EDGE_THRESHOLD_PCT"
    --hub-selection-metric "$HUB_SELECTION_METRIC"
)

if [[ -n "${SUBJECTS}" ]]; then
    # shellcheck disable=SC2206
    SUBJECT_ARRAY=(${SUBJECTS})
    CMD+=(--subjects "${SUBJECT_ARRAY[@]}")
fi

if [[ -n "$CONTAINER" && -f "$CONTAINER" ]]; then
    export APPTAINER_BIND="$BIND_PATHS"
    echo "Running in container: ${CONTAINER}"
    srun apptainer exec "$CONTAINER" "${CMD[@]}"
else
    echo "Container not found at '$CONTAINER'; running on host environment"
    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: 'python' not found on host, and container is unavailable."
        echo "Set CONTAINER to a valid .sif path or load a Python module."
        exit 1
    fi
    "${CMD[@]}"
fi

echo "Group aggregation: complete"

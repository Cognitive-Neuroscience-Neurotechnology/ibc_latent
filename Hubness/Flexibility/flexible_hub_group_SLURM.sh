#!/bin/bash

#SBATCH --job-name=group-flex-hub
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output/%A_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors/%A_%x_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub_group_SLURM.sh

set -euo pipefail

SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub_group.py"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
ASSIGNMENT_DIR="${ASSIGNMENT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
NETWORK_LABEL_BASE="${NETWORK_LABEL_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks}"

ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-network}"
EDGE_THRESHOLD_PCT="${EDGE_THRESHOLD_PCT:-80}"
HUB_SELECTION_METRIC="${HUB_SELECTION_METRIC:-gvc}"
SUBJECTS_FILE="${SUBJECTS_FILE:-}"

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors

echo "=========================================="
echo "Flexible Hub Group Aggregation"
echo "=========================================="
echo "Analysis level: ${ANALYSIS_LEVEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Edge Threshold Percentile: ${EDGE_THRESHOLD_PCT}"
echo "Hub Selection Metric: ${HUB_SELECTION_METRIC}"
echo "Container: ${CONTAINER}"
echo "=========================================="

CMD=(
    python "$SCRIPT"
    --analysis-level "$ANALYSIS_LEVEL"
    --output-dir "$OUTPUT_DIR"
    --assignment-dir "$ASSIGNMENT_DIR"
    --network-label-base "$NETWORK_LABEL_BASE"
    --edge-threshold-percentile "$EDGE_THRESHOLD_PCT"
    --hub-selection-metric "$HUB_SELECTION_METRIC"
)

if [[ -n "$SUBJECTS_FILE" && -f "$SUBJECTS_FILE" ]]; then
    mapfile -t SUBJECTS < "$SUBJECTS_FILE"
    if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
        CMD+=(--subjects "${SUBJECTS[@]}")
    fi
fi

if [[ -n "$CONTAINER" && -f "$CONTAINER" ]]; then
    export APPTAINER_BIND="$BIND_PATHS"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    echo "Running in container: ${CONTAINER}"
    srun apptainer exec "$CONTAINER" "${CMD[@]}"
else
    echo "Container not found at '${CONTAINER}'; running on host environment"
    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: 'python' not found on host, and container is unavailable."
        echo "Set CONTAINER to a valid .sif path or load a Python module."
        exit 1
    fi
    "${CMD[@]}"
fi
#!/bin/bash

#SBATCH --job-name=flex-hub
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -euo pipefail

SUBJECTS_FILE="${SUBJECTS_FILE:-/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt}"
SUBJECT_LINE=$((SLURM_ARRAY_TASK_ID + 1))
SUBJECT=$(sed -n "${SUBJECT_LINE}p" "$SUBJECTS_FILE")
SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub.py"

OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
ASSIGNMENT_DIR="${ASSIGNMENT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
NETWORK_LABEL_BASE="${NETWORK_LABEL_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks}"
ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-network}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.30}"
EDGE_THRESHOLD_PCT="${EDGE_THRESHOLD_PCT:-95}"
HUB_SELECTION_METRIC="${HUB_SELECTION_METRIC:-gvc}"

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors

echo "=========================================="
echo "Flexible Hub Metrics/Plots (SLURM array)"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "Analysis level: ${ANALYSIS_LEVEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Edge Threshold Percentile: ${EDGE_THRESHOLD_PCT}"
echo "Hub Selection Metric: ${HUB_SELECTION_METRIC}"
echo "Container: ${CONTAINER}"
echo "=========================================="

CMD=(
    python "$SCRIPT"
    --subjects "$SUBJECT"
    --analysis-level "$ANALYSIS_LEVEL"
    --output-dir "$OUTPUT_DIR"
    --assignment-dir "$ASSIGNMENT_DIR"
    --network-label-base "$NETWORK_LABEL_BASE"
    --edge-threshold-percentile "$EDGE_THRESHOLD_PCT"
    --hub-selection-metric "$HUB_SELECTION_METRIC"
)

if [[ "$ANALYSIS_LEVEL" == "network_parcel" ]]; then
    CMD+=(--overlap-threshold "$OVERLAP_THRESHOLD")
fi

srun --container-image="$CONTAINER" --container-mounts="$BIND_PATHS" "${CMD[@]}"

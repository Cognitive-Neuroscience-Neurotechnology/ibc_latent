#!/bin/bash

#SBATCH --job-name=aggregate-flex-hub
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=2,5
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub_SLURM.sh

set -euo pipefail

SUBJECTS_FILE="${SUBJECTS_FILE:-/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt}"
SUBJECT_LINE=$((SLURM_ARRAY_TASK_ID + 1))
SUBJECT=$(sed -n "${SUBJECT_LINE}p" "$SUBJECTS_FILE")
#SUBJECT=04

SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub.py"

OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
ASSIGNMENT_DIR="${ASSIGNMENT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
NETWORK_LABEL_BASE="${NETWORK_LABEL_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks}"

ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-network_parcel}" # network or network_parcel (if network_parcel -> also specify OVERLAP_THRESHOLD)
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.30}"
EDGE_THRESHOLD_PCT="${EDGE_THRESHOLD_PCT:-99.7}"
HUB_SELECTION_METRIC="${HUB_SELECTION_METRIC:-gvc}"
SPRING_K="${SPRING_K:-4.0}"
SPRING_ITERATIONS="${SPRING_ITERATIONS:-600}"
SPRING_SCALE="${SPRING_SCALE:-6.0}"
SPRING_MAX_LABELS="${SPRING_MAX_LABELS:-12}"
 
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
echo "Spring K: ${SPRING_K}"
echo "Spring Iterations: ${SPRING_ITERATIONS}"
echo "Spring Scale: ${SPRING_SCALE}"
echo "Spring Max Labels: ${SPRING_MAX_LABELS}"
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
    --spring-k "$SPRING_K"
    --spring-iterations "$SPRING_ITERATIONS"
    --spring-scale "$SPRING_SCALE"
    --spring-max-labels "$SPRING_MAX_LABELS"
)

if [[ "$ANALYSIS_LEVEL" == "network_parcel" ]]; then
    CMD+=(--overlap-threshold "$OVERLAP_THRESHOLD")
fi

if [[ -n "$CONTAINER" && -f "$CONTAINER" ]]; then
    export APPTAINER_BIND="$BIND_PATHS"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
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

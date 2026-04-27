#!/bin/bash

#SBATCH --job-name=static-hubs
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Compute resting-state FC and hub metrics for Glasser 360 or split network parcels.

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Hubness/static_hubs_SLURM.sh

# Environment variable options (set before sbatch to override):
#   ANALYSIS_LEVEL=network_parcel - Analysis level: network | network_parcel
#   FPN_MODE=split          - FPN handling: unified (default) | split
#   SPLIT_PARCELS=1          - Legacy alias for ANALYSIS_LEVEL=network_parcel
#   OVERLAP_THRESHOLD=0.50   - Overlap threshold for split parcels (default: 0.30)
#   EDGE_THRESHOLD_PCT=98    - Percentile threshold for spring plot edges (default: 98)
#   TOP_HUBS_K=10            - Number of top hubs to label/export (default: 10)
#   HUB_SELECTION_METRIC=strength - Hub ranking metric: strength | participation
#   SAVE_NETWORK_FC=1        - Save network-collapsed FC NPZ (needed for group aggregation script)
#   SAVE_TOP_HUBS_DLABEL=1   - Save top-hubs dlabel for wb_view when using split parcels
#   PLOT_ONLY_FROM_SAVED_FC=1 - Regenerate spring plot from saved FC npz (skip FC recomputation)

set -euo pipefail

SUBJECTS_FILE="${SUBJECTS_FILE:-/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt}"
SUBJECT_LINE=$((SLURM_ARRAY_TASK_ID + 1))
SUBJECT=$(sed -n "${SUBJECT_LINE}p" "$SUBJECTS_FILE")
#SUBJECT=04
SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/static_hubs.py"

FMRIPREP_BASE="${FMRIPREP_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out}"
ASSIGNMENT_DIR="${ASSIGNMENT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
PARCELLATION_PATH="${PARCELLATION_PATH:-}"

# ---PARAMETERS------------------------------------------------------------
ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-network}"
FPN_MODE="${FPN_MODE:-unified}"
SPLIT_PARCELS="${SPLIT_PARCELS:-1}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.30}"
EDGE_THRESHOLD_PCT="${EDGE_THRESHOLD_PCT:-50}"
TOP_HUBS_K="${TOP_HUBS_K:-15}"
HUB_SELECTION_METRIC="${HUB_SELECTION_METRIC:-strength}"
SAVE_NETWORK_FC="${SAVE_NETWORK_FC:-1}"
SAVE_TOP_HUBS_DLABEL="${SAVE_TOP_HUBS_DLABEL:-1}"
# Use this flag to skip FC recomputation and just regenerate the spring plot from the saved FC npz file (useful for tweaking plot parameters without rerunning the whole analysis)
PLOT_ONLY_FROM_SAVED_FC="${PLOT_ONLY_FROM_SAVED_FC:-0}"
# --------------------------------------------------------------------------

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_static_logs/errors

if [[ -z "$ANALYSIS_LEVEL" ]]; then
    if [[ "$SPLIT_PARCELS" == "1" ]]; then
        ANALYSIS_LEVEL="network_parcel"
    else
        ANALYSIS_LEVEL="network"
    fi
fi

echo "=========================================="
echo "Static Hubs FC (SLURM array)"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-single}/${SLURM_ARRAY_TASK_MAX:-single}"
echo "Script: ${SCRIPT}"
echo "Analysis level: ${ANALYSIS_LEVEL}"
echo "FPN mode: ${FPN_MODE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Edge Threshold Percentile: ${EDGE_THRESHOLD_PCT}"
echo "Hub Selection Metric: ${HUB_SELECTION_METRIC}"
echo "Save Network FC: ${SAVE_NETWORK_FC}"
if [ "$ANALYSIS_LEVEL" = "network_parcel" ]; then
    echo "Overlap Threshold: ${OVERLAP_THRESHOLD}"
    echo "Top Hubs K: ${TOP_HUBS_K}"
    echo "Save Top Hubs Dlabel: ${SAVE_TOP_HUBS_DLABEL}"
    echo "Plot Only From Saved FC: ${PLOT_ONLY_FROM_SAVED_FC}"
fi
echo "Container: ${CONTAINER}"
echo "=========================================="

CMD=(
    "python" "$SCRIPT"
    --subjects "$SUBJECT"
    --fmriprep-base "$FMRIPREP_BASE"
    --assignment-dir "$ASSIGNMENT_DIR"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$PARCELLATION_PATH" ]]; then
    CMD+=(--parcellation-path "$PARCELLATION_PATH")
fi

CMD+=(--analysis-level "$ANALYSIS_LEVEL")
CMD+=(--fpn-mode "$FPN_MODE")
CMD+=(--edge-threshold-percentile "$EDGE_THRESHOLD_PCT")
CMD+=(--hub-selection-metric "$HUB_SELECTION_METRIC")

if [[ "$SAVE_NETWORK_FC" == "1" || "$SAVE_NETWORK_FC" == "true" || "$SAVE_NETWORK_FC" == "TRUE" ]]; then
    CMD+=(--save-network-fc)
fi

if [[ "$ANALYSIS_LEVEL" == "network_parcel" ]]; then
    CMD+=(--overlap-threshold "$OVERLAP_THRESHOLD")
    CMD+=(--top-hubs-k "$TOP_HUBS_K")
    if [[ "$PLOT_ONLY_FROM_SAVED_FC" == "1" || "$PLOT_ONLY_FROM_SAVED_FC" == "true" || "$PLOT_ONLY_FROM_SAVED_FC" == "TRUE" ]]; then
        CMD+=(--plot-only-from-saved-fc)
    fi
    if [[ "$SAVE_TOP_HUBS_DLABEL" == "1" || "$SAVE_TOP_HUBS_DLABEL" == "true" || "$SAVE_TOP_HUBS_DLABEL" == "TRUE" ]]; then
        CMD+=(--save-top-hubs-dlabel)
    fi
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

echo "Subject ${SUBJECT}: complete"

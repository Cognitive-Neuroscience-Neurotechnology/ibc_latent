#!/bin/bash

#SBATCH --job-name=flex-hub-ppi
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -euo pipefail

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub_ppi_SLURM.sh

SUBJECTS_FILE="${SUBJECTS_FILE:-/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt}"
SUBJECT_LINE=$((SLURM_ARRAY_TASK_ID + 1))
SUBJECT=$(sed -n "${SUBJECT_LINE}p" "$SUBJECTS_FILE")

SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/Flexibility/flexible_hub_ppi.py"

FMRIPREP_BASE="${FMRIPREP_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out}"
ASSIGNMENT_DIR="${ASSIGNMENT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
NETWORK_LABEL_BASE="${NETWORK_LABEL_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks}"
PARCELLATION_PATH="${PARCELLATION_PATH:-}"

ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-network}" # network or network_parcel
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.30}" # Only used for network_parcel level (default: 0.30, alternative: 0.50)

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_flex_logs/errors

echo "=========================================="
echo "Flexible Hub PPI (SLURM array)"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "Analysis level: ${ANALYSIS_LEVEL}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Container: ${CONTAINER}"
echo "=========================================="

CMD=(
    python "$SCRIPT"
    --subjects "$SUBJECT"
    --analysis-level "$ANALYSIS_LEVEL"
    --fmriprep-base "$FMRIPREP_BASE"
    --assignment-dir "$ASSIGNMENT_DIR"
    --network-label-base "$NETWORK_LABEL_BASE"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$PARCELLATION_PATH" ]]; then
    CMD+=(--parcellation-path "$PARCELLATION_PATH")
fi

if [[ "$ANALYSIS_LEVEL" == "network_parcel" ]]; then
    CMD+=(--overlap-threshold "$OVERLAP_THRESHOLD")
fi

if [[ -n "$CONTAINER" && -f "$CONTAINER" ]]; then
    export APPTAINER_BIND="$BIND_PATHS"
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

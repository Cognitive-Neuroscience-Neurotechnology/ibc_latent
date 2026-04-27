#!/bin/bash

#SBATCH --job-name=define-networks
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/hubness_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Split Glasser parcels by overlapping network labels per subject using SLURM array.
# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Hubness/define_networks_SLURM.sh

set -euo pipefail

SUBJECTS_FILE="${SUBJECTS_FILE:-/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt}"
SUBJECT_LINE=$((SLURM_ARRAY_TASK_ID + 1))
SUBJECT=$(sed -n "${SUBJECT_LINE}p" "$SUBJECTS_FILE")
#SUBJECT=04
SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Hubness/define_networks.py"

NETWORK_LABEL_BASE="${NETWORK_LABEL_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks}"
PARCELIZED_FPN_BASE="${PARCELIZED_FPN_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/parcelized_fpn}"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/hubness}"
PARCELLATION_PATH="${PARCELLATION_PATH:-}"

# ---PARAMETERS------------------------------------------------------------
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.30}" # change threshold
FPN_MODE="${FPN_MODE:-unified}" # options: unified (default), split
DLABEL_COLORING="${DLABEL_COLORING:-both}" # options: 'both' (default), 'network', 'parcel', 'none'
SKIP_DLABEL="${SKIP_DLABEL:-0}" # set to 1 to skip dlabel creation (e.g. for testing or if only network outputs are needed)
DLABEL_ONLY="${DLABEL_ONLY:-0}" # set to 1 to only create dlabel outputs (skips network and parcel outputs; use if only dlabels are needed)
# --------------------------------------------------------------------------

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/hubness_logs/errors

echo "=========================================="
echo "Define Networks (SLURM array)"
echo "=========================================="
echo "Subject: ${SUBJECT}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-single}/${SLURM_ARRAY_TASK_MAX:-single}"
echo "Script: ${SCRIPT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Overlap Threshold: ${OVERLAP_THRESHOLD}"
echo "FPN mode: ${FPN_MODE}"
echo "Dlabel coloring: ${DLABEL_COLORING}"
echo "Dlabel only mode: ${DLABEL_ONLY}"
echo "Skip dlabel creation: ${SKIP_DLABEL}"
echo "Container: ${CONTAINER}"
echo "=========================================="

CMD=(
    "python" "$SCRIPT"
    --subjects "$SUBJECT"
    --network-label-base "$NETWORK_LABEL_BASE"
    --parcelized-fpn-base "$PARCELIZED_FPN_BASE"
    --output-dir "$OUTPUT_DIR"
    --overlap-threshold "$OVERLAP_THRESHOLD"
    --fpn-mode "$FPN_MODE"
    --dlabel-coloring "$DLABEL_COLORING"
)

if [[ -n "$PARCELLATION_PATH" ]]; then
    CMD+=(--parcellation-path "$PARCELLATION_PATH")
fi

if [[ "$DLABEL_ONLY" == "1" ]]; then
    CMD+=(--dlabel-only)
fi

if [[ "$SKIP_DLABEL" == "1" ]]; then
    CMD+=(--skip-dlabel)
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

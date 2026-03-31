#!/bin/bash

#SBATCH --job-name=fpn_parcels
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/mask_to_parcel/fpn_parcels_%j.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/mask_to_parcel/fpn_parcels_%j.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# SLURM script for parcelizing individual FPN masks (FPN/FPN_A/FPN_B)
# Submit with: sbatch Subnetworks/Subnetwork_Derivation/fpn_masks_to_parcels_SLURM.sh

set -euo pipefail

echo "============================================"
echo "FPN Masks to Parcels (SLURM)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Paths
SCRIPT_DIR="/home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation"
CONFIG_FILE="/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"
SUBNETWORK_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
OUTPUT_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/parcelized_fpn"

# Runtime options (override via env at submit time)
# Example:
# OVERLAP_THRESHOLD=0.5 K_INDEX=0 sbatch Subnetworks/Subnetwork_Derivation/fpn_masks_to_parcels_SLURM.sh
K_INDEX="${K_INDEX:-0}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.50}"
PARCELLATION_PATH="${PARCELLATION_PATH:-}"

# Container
container=/home/rglz/containers/gfae.sif
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Read subjects from config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
SUBJECTS=$(awk '{print $1}' "$CONFIG_FILE" | tr '\n' ' ')

if [[ -z "${SUBJECTS// }" ]]; then
    echo "ERROR: No subjects found in $CONFIG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

echo "Subnetwork directory: $SUBNETWORK_DIR"
echo "Output directory:     $OUTPUT_DIR"
echo "Subjects:             $SUBJECTS"
echo "K index:              $K_INDEX"
echo "Overlap threshold:    $OVERLAP_THRESHOLD"
if [[ -n "$PARCELLATION_PATH" ]]; then
    echo "Parcellation path:    $PARCELLATION_PATH"
else
    echo "Parcellation path:    auto-detect in script"
fi
echo "Container:            $container"
echo ""

cmd=(python "$SCRIPT_DIR/fpn_masks_to_parcels.py"
    --subjects $SUBJECTS
    --subnetwork-dir "$SUBNETWORK_DIR"
    --output "$OUTPUT_DIR"
    --k-index "$K_INDEX"
    --overlap-threshold "$OVERLAP_THRESHOLD"
)

if [[ -n "$PARCELLATION_PATH" ]]; then
    cmd+=(--parcellation-path "$PARCELLATION_PATH")
fi

echo "============================================"
echo "Running FPN parcelization"
echo "============================================"
printf 'Command: '
printf '%q ' "${cmd[@]}"
echo ""

srun apptainer exec "${container}" "${cmd[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Completed successfully!"
    echo "Results: $OUTPUT_DIR"
else
    echo "FAILED with exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "============================================"

exit $EXIT_CODE

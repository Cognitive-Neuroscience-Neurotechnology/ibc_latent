#!/bin/bash

#SBATCH --job-name=fpn_overlap
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/subnetwork_logs/output/%j_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/subnetwork_logs/errors/%j_%x_%u.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Build across-subject FPN overlap heatmap dscalars.
# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/fpn_overlap_heatmap_SLURM.sh

set -euo pipefail

SCRIPT="/home/hmueller2/ibc_code/ibc_latent/Subnetworks/fpn_overlap_heatmap.py"
SUBJECTS_FILE="${SUBJECTS_FILE:-/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt}"
SUBNETWORK_BASE="${SUBNETWORK_BASE:-/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap}"
OUTPUT_DIR="${OUTPUT_DIR:-/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/fpn_overlap_heatmap}"
MODE="${MODE:-both}"                     # whole | subnetworks | both
K_INDEX="${K_INDEX:-0}"
INPUT_NAME="${INPUT_NAME:-}"
FPNA_LABEL="${FPNA_LABEL:-}"
FPNB_LABEL="${FPNB_LABEL:-}"
EXPECTED_SUBJECTS="${EXPECTED_SUBJECTS:-8}"

CONTAINER="${CONTAINER:-/home/rglz/containers/gfae.sif}"
BIND_PATHS="${BIND_PATHS:-/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2}"

mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/subnetwork_logs/output
mkdir -p /ptmp/hmueller2/2025_ibc_latent/logs/subnetwork_logs/errors
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "ERROR: subjects file not found: $SUBJECTS_FILE"
    exit 1
fi

SUBJECTS=$(awk '{print $1}' "$SUBJECTS_FILE" | tr '\n' ' ')
if [[ -z "${SUBJECTS// }" ]]; then
    echo "ERROR: no subjects parsed from $SUBJECTS_FILE"
    exit 1
fi

echo "=========================================="
echo "FPN Overlap Heatmap (SLURM)"
echo "=========================================="
echo "Script:           $SCRIPT"
echo "Subjects file:    $SUBJECTS_FILE"
echo "Subnetwork base:  $SUBNETWORK_BASE"
echo "Output dir:       $OUTPUT_DIR"
echo "Mode:             $MODE"
echo "K index:          $K_INDEX"
echo "Expected subjects:$EXPECTED_SUBJECTS"
echo "Container:        $CONTAINER"
echo "=========================================="

CMD=(
    python "$SCRIPT"
    --subjects $SUBJECTS
    --subnetwork-base "$SUBNETWORK_BASE"
    --output-dir "$OUTPUT_DIR"
    --mode "$MODE"
    --k-index "$K_INDEX"
    --expected-subjects "$EXPECTED_SUBJECTS"
)

if [[ -n "$INPUT_NAME" ]]; then
    CMD+=(--input-name "$INPUT_NAME")
fi

if [[ -n "$FPNA_LABEL" ]]; then
    CMD+=(--fpna-label "$FPNA_LABEL")
fi

if [[ -n "$FPNB_LABEL" ]]; then
    CMD+=(--fpnb-label "$FPNB_LABEL")
fi

printf 'Command: %q ' "${CMD[@]}"
echo ""

if [[ -n "$CONTAINER" && -f "$CONTAINER" ]]; then
    export APPTAINER_BIND="$BIND_PATHS"
    srun apptainer exec "$CONTAINER" "${CMD[@]}"
else
    echo "Container not found at '$CONTAINER'; running on host"
    "${CMD[@]}"
fi

echo "Completed: FPN overlap heatmaps"

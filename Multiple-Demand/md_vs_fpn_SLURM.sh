#!/bin/bash

#SBATCH --job-name=md_fpn
#SBATCH --output=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_fpn_%j.out
#SBATCH --error=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_fpn_%j.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4


# SLURM script for runningcomparing MD maps and FPN in individuals
# Submit with: sbatch Multiple-Demand/md_vs_fpn_SLURM.sh

set -euo pipefail

echo "============================================"
echo "MD vs FPN Mapping (SLURM)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Paths
SCRIPT_DIR="/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand"
CONFIG_FILE="/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"
MD_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise"
SUBNETWORK_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
OUTPUT_DIR="${MD_DIR}/md_vs_fpn"

THRESHOLD_PERCENT="${THRESHOLD_PERCENT:-20}"

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

echo "MD directory:         $MD_DIR"
echo "Subnetwork directory: $SUBNETWORK_DIR"
echo "Output directory:     $OUTPUT_DIR"
echo "Subjects:             $SUBJECTS"
echo "Threshold percent:    top $THRESHOLD_PERCENT%"
echo "Container:            $container"
echo ""

cmd=(python "$SCRIPT_DIR/md_vs_fpn.py"
    --subjects $SUBJECTS
    --md-dir "$MD_DIR"
    --subnetwork-dir "$SUBNETWORK_DIR"
    --output "$OUTPUT_DIR"
    --threshold-percent "$THRESHOLD_PERCENT"
)

echo "============================================"
echo "Running MD vs FPN comparison"
echo "============================================"
printf 'Command: %q ' "${cmd[@]}"
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
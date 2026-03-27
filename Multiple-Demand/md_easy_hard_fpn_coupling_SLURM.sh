#!/bin/bash

#SBATCH --job-name=md_easy_hard
#SBATCH --output=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_easy_hard_%j.out
#SBATCH --error=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_easy_hard_%j.err
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# SLURM script for Easy/Hard activation and FPN-DMN/DAN coupling analysis
# Submit with: sbatch Multiple-Demand/md_easy_hard_fpn_coupling_SLURM.sh

set -euo pipefail

echo "============================================"
echo "MD Easy/Hard FPN Coupling (SLURM)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Paths
SCRIPT_DIR="/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand"
CONFIG_FILE="/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"
CONTRAST_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR"
DTSERIES_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out"
EVENTS_BASE="/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw"
SUBNETWORK_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
NETWORK_LABEL_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks"
ALL_CONTRASTS_TSV="/home/hmueller2/ibc_code/ibc_latent/Data Info/all_contrasts.tsv"
OUTPUT_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise/md_easy_hard_fpn_coupling"

K_INDEX="${K_INDEX:-0}"
STRICT="${STRICT:-0}"

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

echo "Contrast base:      $CONTRAST_BASE"
echo "Dtseries base:      $DTSERIES_BASE"
echo "Events base:        $EVENTS_BASE"
echo "Subnetwork dir:     $SUBNETWORK_DIR"
echo "Network label base: $NETWORK_LABEL_BASE"
echo "Output directory:   $OUTPUT_DIR"
echo "Subjects:           $SUBJECTS"
echo "K index:            $K_INDEX"
echo "Strict mode:        $STRICT"
echo "Container:          $container"
echo ""

cmd=(python "$SCRIPT_DIR/md_easy_hard_fpn_coupling.py"
    --subjects $SUBJECTS
    --contrast-base "$CONTRAST_BASE"
    --dtseries-base "$DTSERIES_BASE"
    --events-base "$EVENTS_BASE"
    --subnetwork-dir "$SUBNETWORK_DIR"
    --network-label-base "$NETWORK_LABEL_BASE"
    --all-contrasts-tsv "$ALL_CONTRASTS_TSV"
    --output "$OUTPUT_DIR"
    --k-index "$K_INDEX"
)

if [ "$STRICT" = "1" ]; then
    cmd+=(--strict)
fi

echo "============================================"
echo "Running Easy/Hard FPN coupling analysis"
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
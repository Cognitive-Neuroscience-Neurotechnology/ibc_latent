#!/bin/bash
#SBATCH --job-name=md_group
#SBATCH --output=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_group_%j.out
#SBATCH --error=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_group_%j.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

#
# SLURM script for computing group-level MD map
#
# Run this after md_mapping_SLURM_array.sh completes
#
# Submit with: sbatch md_mapping_SLURM_group.sh
#

echo "============================================"
echo "Multiple Demand System - Group Analysis"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set your paths here
CONTRAST_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR"
OUTPUT_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/md_system"
CONFIG_FILE="/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"

# Read subjects from config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
SUBJECTS=$(awk '{print $1}' "$CONFIG_FILE" | tr '\n' ' ')

echo "Contrast base: $CONTRAST_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Subjects: $SUBJECTS"
echo ""

# Script directory (absolute path)
SCRIPT_DIR="/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# Container setup (using apptainer) - Update with your container path
container=/home/rglz/containers/gfae.sif  # Change to your container path
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Check if container exists
if [ ! -f "$container" ]; then
    echo "ERROR: Container not found: $container"
    exit 1
fi

echo "Script directory: $SCRIPT_DIR"
echo "Container: $container"
echo ""

# Check if Python script exists
if [ ! -f "$SCRIPT_DIR/md_mapping.py" ]; then
    echo "ERROR: md_mapping.py not found in $SCRIPT_DIR"
    exit 1
fi

# Compute group map
echo "Computing group-level MD map..."
echo ""

srun apptainer exec ${container} python "$SCRIPT_DIR/md_mapping.py" \
    --subjects $SUBJECTS \
    --group \
    --contrast-base "$CONTRAST_BASE" \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Group analysis completed successfully!"
    echo "============================================"
    echo "Results saved to: $OUTPUT_DIR/group/"
    echo ""
    echo "To view results:"
    echo "  wb_view $OUTPUT_DIR/group/group_MD_mean.dscalar.nii"
    echo ""
    echo "To compare subjects:"
    echo "  python visualize_md_maps.py compare $OUTPUT_DIR --output $OUTPUT_DIR/figures"
else
    echo "Group analysis failed with exit code: $EXIT_CODE"
    echo "============================================"
fi

echo ""
echo "Job finished at: $(date)"
echo "============================================"

exit $EXIT_CODE

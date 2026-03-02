#!/bin/bash
#SBATCH --job-name=md_mapping_array
#SBATCH --output=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_mapping_sub-%a_%j.out
#SBATCH --error=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_mapping_sub-%a_%j.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-7


# SLURM array job script for running MD mapping per subject

# This processes each subject in parallel as a separate job.
# After all subjects complete, run md_mapping_SLURM_group.sh to compute group map.

# Submit with: sbatch md_mapping_SLURM_array.sh


# Config file with subjects
CONFIG_FILE="/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"

# Get subject for this array task
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_FILE")
SUBJECT=$(echo "$line" | awk '{print $1}')

echo "============================================"
echo "Multiple Demand System Mapping - Subject $SUBJECT"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set your paths here
CONTRAST_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR"
OUTPUT_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/md_system"

echo "Subject: $SUBJECT"
echo "Contrast base: $CONTRAST_BASE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Script directory (absolute path)
SCRIPT_DIR="/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand"

# Container setup (using apptainer) - Update with your container path
container=/home/rglz/containers/gfae.sif  # Change to your container path
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

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

# Process this subject
echo "Processing subject $SUBJECT..."
echo ""

srun apptainer exec ${container} python "$SCRIPT_DIR/md_mapping.py" \
    --subject "$SUBJECT" \
    --contrast-base "$CONTRAST_BASE" \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Subject $SUBJECT completed successfully!"
else
    echo "Subject $SUBJECT failed with exit code: $EXIT_CODE"
fi
echo "Job finished at: $(date)"
echo "============================================"

exit $EXIT_CODE

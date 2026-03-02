#!/bin/bash
#
# Example script for running MD mapping analysis on IBC dataset
#
# Usage: bash run_md_mapping.sh

# Set your paths here
CONTRAST_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR"
OUTPUT_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/md_system"

# List of IBC subjects (adjust based on your data)
SUBJECTS=$(cat /2025_ibc_latent/misc/subjects_resting.txt)

echo "============================================"
echo "Multiple Demand System Mapping"
echo "============================================"
echo ""
echo "Contrast base: $CONTRAST_BASE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Option 1: Process all subjects individually
echo "Processing individual subjects..."
for SUBJECT in $SUBJECTS; do
    echo ""
    echo "--- Subject $SUBJECT ---"
    python md_mapping.py \
        --subject "$SUBJECT" \
        --contrast-base "$CONTRAST_BASE" \
        --output "$OUTPUT_DIR"
done

# Option 2: Process all subjects at once with group analysis
echo ""
echo "============================================"
echo "Computing group-level MD map..."
echo "============================================"
python md_mapping.py \
    --subjects $SUBJECTS \
    --group \
    --contrast-base "$CONTRAST_BASE" \
    --output "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "Analysis complete!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  wb_view $OUTPUT_DIR/group/group_MD_mean.dscalar.nii"

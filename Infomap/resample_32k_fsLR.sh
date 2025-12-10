#!/bin/bash
# filepath: /home/hmueller2/ibc_code/ibc_latent/Infomap/resample_32k_fsLR.sh

#SBATCH --job-name=resample_32k
#SBATCH --output=/ptmp/hmueller2/pipeline_logs/output/resample_32k_%A_%a.out
#SBATCH --error=/ptmp/hmueller2/pipeline_logs/errors/resample_32k_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=thin
#SBATCH --mem=4G
# #SBATCH --array=0-7

# This script resamples native cortical surfaces to the fsLR 32k surface space.
# INPUT:
# - Native surface files (e.g., midthickness surfaces) in GIFTI format
# - Registration spheres for fsLR alignment
# - Template fsLR 32k spheres from HCP
# OUTPUT:
# - Resampled surfaces in fsLR 32k space in GIFTI format

# Get subject from subjects file
#SUBJECTS_FILE="/ptmp/hmueller2/Downloads/subjects_resting.txt"
#SUB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SUBJECTS_FILE)
SUB="06"  # Manually set subject ID here

echo "Processing subject: $SUB"

# Set paths
WB_CMD="/home/hmueller2/workbench/bin_linux64/wb_command"
IN_DIR="/ptmp/hmueller2/Downloads/fmriprep_out/sub-${SUB}/anat"
OUT_DIR="$IN_DIR"

# Verify input directory exists
if [[ ! -d "$IN_DIR" ]]; then
    echo "ERROR: Input directory does not exist: $IN_DIR"
    exit 1
fi

# Template fsLR 32k spheres (Workbench)
FS32K_L="/ptmp/hmueller2/Downloads/hcp/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii"
FS32K_R="/ptmp/hmueller2/Downloads/hcp/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii"

# Verify template files exist
if [[ ! -f "$FS32K_L" || ! -f "$FS32K_R" ]]; then
    echo "ERROR: Template fsLR 32k spheres not found"
    exit 1
fi

# Registration spheres
REG_L="${IN_DIR}/sub-${SUB}_hemi-L_space-fsLR_desc-reg_sphere.surf.gii"
REG_R="${IN_DIR}/sub-${SUB}_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"

# Only resample midthickness surface
surf="midthickness"
NATIVE_L="${IN_DIR}/sub-${SUB}_hemi-L_${surf}.surf.gii"
NATIVE_R="${IN_DIR}/sub-${SUB}_hemi-R_${surf}.surf.gii"
OUT_L="${OUT_DIR}/sub-${SUB}_hemi-L_${surf}.32k_fs_LR.surf.gii"
OUT_R="${OUT_DIR}/sub-${SUB}_hemi-R_${surf}.32k_fs_LR.surf.gii"

# Left hemisphere
if [[ -f "$NATIVE_L" && -f "$REG_L" && -f "$FS32K_L" ]]; then
    echo "Resampling left hemisphere: $NATIVE_L -> $OUT_L"
    "$WB_CMD" -surface-resample "$NATIVE_L" "$REG_L" "$FS32K_L" BARYCENTRIC "$OUT_L"
    echo "✓ Left hemisphere complete"
else
    echo "ERROR: Missing file(s) for left hemisphere:"
    [[ ! -f "$NATIVE_L" ]] && echo "  - Native surface: $NATIVE_L"
    [[ ! -f "$REG_L" ]] && echo "  - Registration sphere: $REG_L"
    [[ ! -f "$FS32K_L" ]] && echo "  - Template sphere: $FS32K_L"
    exit 1
fi

# Right hemisphere
if [[ -f "$NATIVE_R" && -f "$REG_R" && -f "$FS32K_R" ]]; then
    echo "Resampling right hemisphere: $NATIVE_R -> $OUT_R"
    "$WB_CMD" -surface-resample "$NATIVE_R" "$REG_R" "$FS32K_R" BARYCENTRIC "$OUT_R"
    echo "✓ Right hemisphere complete"
else
    echo "ERROR: Missing file(s) for right hemisphere:"
    [[ ! -f "$NATIVE_R" ]] && echo "  - Native surface: $NATIVE_R"
    [[ ! -f "$REG_R" ]] && echo "  - Registration sphere: $REG_R"
    [[ ! -f "$FS32K_R" ]] && echo "  - Template sphere: $FS32K_R"
    exit 1
fi

echo "✓ Surface resampling complete for sub-${SUB}"

# Compute vertex area for left hemisphere
echo "Computing vertex area for left hemisphere..."
L_VA="${OUT_DIR}/sub-${SUB}_hemi-L_midthickness_va.32k_fs_LR.shape.gii"
"$WB_CMD" -surface-vertex-areas "$OUT_L" "$L_VA"

# Compute vertex area for right hemisphere
echo "Computing vertex area for right hemisphere..."
R_VA="${OUT_DIR}/sub-${SUB}_hemi-R_midthickness_va.32k_fs_LR.shape.gii"
"$WB_CMD" -surface-vertex-areas "$OUT_R" "$R_VA"

# Create CIFTI dscalar from left and right vertex area files
echo "Creating CIFTI vertex area file..."
CIFTI_VA="${OUT_DIR}/sub-${SUB}.midthickness_va.32k_fs_LR.dscalar.nii"
"$WB_CMD" -cifti-create-dense-scalar \
    "$CIFTI_VA" \
    -left-metric "$L_VA" \
    -right-metric "$R_VA"

echo "✓ Vertex area computation complete: $CIFTI_VA"

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Infomap/resample_32k_fsLR.sh
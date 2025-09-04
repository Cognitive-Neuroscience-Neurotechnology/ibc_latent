#!/bin/bash
# filepath: /home/hmueller2/ibc_code/ibc_latent/Infomap/generate_vertex_area.sh

# make executable: chmod +x /home/hmueller2/ibc_code/ibc_latent/Infomap/generate_vertex_area.sh
# run: /home/hmueller2/ibc_code/ibc_latent/Infomap/generate_vertex_area.sh

# Set base directory
BASE_DIR="/ptmp/hmueller2/Downloads/fmriprep_out"

# Loop over subjects 01 to 15
for i in 01 04 05 06 07 08 09 11 12 13 14 15; do
    SUBJ="sub-${i}"
    ANAT_DIR="${BASE_DIR}/${SUBJ}/anat"
    L_SURF="${ANAT_DIR}/${SUBJ}_hemi-L_midthickness.32k_fs_LR.surf.gii"
    R_SURF="${ANAT_DIR}/${SUBJ}_hemi-R_midthickness.32k_fs_LR.surf.gii"
    L_VA="${ANAT_DIR}/${SUBJ}_hemi-L_midthickness_va.32k_fs_LR.shape.gii"
    R_VA="${ANAT_DIR}/${SUBJ}_hemi-R_midthickness_va.32k_fs_LR.shape.gii"
    CIFTI_VA="${ANAT_DIR}/${SUBJ}.midthickness_va.32k_fs_LR.dscalar.nii"

    echo "Processing $SUBJ..."

    # Generate vertex area for left hemisphere
    wb_command -surface-vertex-areas "$L_SURF" "$L_VA"

    # Generate vertex area for right hemisphere
    wb_command -surface-vertex-areas "$R_SURF" "$R_VA"

    # Combine into CIFTI dense scalar
    wb_command -cifti-create-dense-scalar "$CIFTI_VA" \
        -left-metric "$L_VA" \
        -right-metric "$R_VA"
done

echo "All subjects processed."
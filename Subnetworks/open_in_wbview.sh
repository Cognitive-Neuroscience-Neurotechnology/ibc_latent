#!/bin/bash

# Usage: ./open_in_wbview.sh 01

SUB=$1

# Base directories
SUBNETWORKS_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/subnetworks/infomap"
FMRIPREP_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/fmriprep_out"

# Files for this subject
DSCALAR="${SUBNETWORKS_DIR}/sub-${SUB}/FPN_communities.dscalar.nii"
SURF_R="${FMRIPREP_DIR}/sub-${SUB}/anat/sub-${SUB}_hemi-R_inflated.32k_fs_LR.surf.gii"
SURF_L="${FMRIPREP_DIR}/sub-${SUB}/anat/sub-${SUB}_hemi-L_inflated.32k_fs_LR.surf.gii"

# Launch wb_view with the three files
wb_view "$DSCALAR" "$SURF_L" "$SURF_R"
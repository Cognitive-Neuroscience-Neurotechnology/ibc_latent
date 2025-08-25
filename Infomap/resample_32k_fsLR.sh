#!/bin/bash
# filepath: /home/hmueller2/ibc_code/ibc_latent/Infomap/resample_surfaces_to_fsLR32k.sh

# Set paths
WB_CMD="/home/hmueller2/workbench/bin_linux64/wb_command"
IN_DIR="/ptmp/hmueller2/Downloads/fmriprep_out/sub-01/anat"
OUT_DIR="$IN_DIR"
SUB="sub-01"

# Template fsLR 32k spheres (Workbench)
FS32K_L="/ptmp/hmueller2/Downloads/hcp/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii"
FS32K_R="/ptmp/hmueller2/Downloads/hcp/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii"

# Registration spheres
REG_L="${IN_DIR}/${SUB}_hemi-L_space-fsLR_desc-reg_sphere.surf.gii"
REG_R="${IN_DIR}/${SUB}_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"

# Only resample midthickness surface
surf="midthickness"
NATIVE_L="${IN_DIR}/${SUB}_hemi-L_${surf}.surf.gii"
NATIVE_R="${IN_DIR}/${SUB}_hemi-R_${surf}.surf.gii"
OUT_L="${OUT_DIR}/${SUB}_hemi-L_${surf}.32k_fs_LR.surf.gii"
OUT_R="${OUT_DIR}/${SUB}_hemi-R_${surf}.32k_fs_LR.surf.gii"

# Left hemisphere
if [[ -f "$NATIVE_L" && -f "$REG_L" && -f "$FS32K_L" ]]; then
  echo "Resampling $NATIVE_L to $OUT_L"
  "$WB_CMD" -surface-resample "$NATIVE_L" "$REG_L" "$FS32K_L" BARYCENTRIC "$OUT_L"
else
  echo "Missing file for left hemisphere: $NATIVE_L, $REG_L, or $FS32K_L"
fi

# Right hemisphere
if [[ -f "$NATIVE_R" && -f "$REG_R" && -f "$FS32K_R" ]]; then
  echo "Resampling $NATIVE_R to $OUT_R"
  "$WB_CMD" -surface-resample "$NATIVE_R" "$REG_R" "$FS32K_R" BARYCENTRIC "$OUT_R"
else
  echo "Missing file for right hemisphere: $NATIVE_R, $REG_R, or $FS32K_R"
fi

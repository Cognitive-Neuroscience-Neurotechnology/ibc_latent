#!/usr/bin/env bash
set -euo pipefail

# --------------------- USER SETTINGS ---------------------
sub="$1"

APPROACH="kmeans"   # Options: "infomap", "kmeans"
SURFACE="inflated"  # Options: "inflated", "very_inflated"
SCENE_NAME="CortexOnly"   # "CortexOnly" for cortex.scene -- "WhiteScene" for template.scene

# Adjust these if needed
WORKING_DIR="/ptmp/hmueller2/Downloads"
SUBNETWORK_DIR="${WORKING_DIR}/subnetworks"
APPROACH_DIR="${SUBNETWORK_DIR}/${APPROACH}"
MASKS_DIR="${WORKING_DIR}/fsLR_masks"
OUT_DIR="${SUBNETWORK_DIR}/brain_plots_k2_wb"
WB_COMMAND="/home/hmueller2/workbench/bin_linux64/wb_command"
WB_VIEW="/home/hmueller2/workbench/bin_linux64/wb_view"

# Add library path for workbench
export LD_LIBRARY_PATH="/home/hmueller2/workbench/libs_linux64:${LD_LIBRARY_PATH:-}"

# --------------------- INPUT FILES (from your message) ---------------------
SURF_L="${MASKS_DIR}/fs_LR.32k.L.${SURFACE}.surf.gii"
SURF_R="${MASKS_DIR}/fs_LR.32k.R.${SURFACE}.surf.gii"
ROI_L="${MASKS_DIR}/L.atlasroi.32k_fs_LR.shape.gii"
ROI_R="${MASKS_DIR}/R.atlasroi.32k_fs_LR.shape.gii"

anat_infomap="${APPROACH_DIR}/sub-${sub}/${sub}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii"

# --------------------- OUTPUT FILES ---------------------
mkdir -p "${OUT_DIR}"
OUT_PREFIX="${OUT_DIR}/sub-${sub}_${APPROACH}"
LEFT_METRIC="${OUT_PREFIX}.L.func.gii"
RIGHT_METRIC="${OUT_PREFIX}.R.func.gii"

# combined CIFTI (optionally rebuilt to match a template)
OUT_CIFTI="${OUT_PREFIX}.dscalar.nii"

# Template 
TEMPLATE_SCENE="${OUT_DIR}/template.scene"   # "cortex.scene" or template.scene
OUT_IMAGE="${OUT_PREFIX}.png"

# --------------------- CHECKS ---------------------
if [ -z "${WB_COMMAND}" ]; then
  echo "ERROR: wb_command not found on PATH. Please install Connectome Workbench and add wb_command to PATH."
  exit 2
fi

if [ ! -f "${anat_infomap}" ]; then
  echo "ERROR: input dscalar.nii not found: ${anat_infomap}"
  exit 3
fi

if [ ! -f "${SURF_L}" ] || [ ! -f "${SURF_R}" ]; then
  echo "ERROR: left/right surfaces not found."
  echo "  SURF_L=${SURF_L}"
  echo "  SURF_R=${SURF_R}"
  exit 4
fi

# --------------------- STEP 1: Extract left & right cortex METRIC files ---------------------
# The dscalar typically contains cortex left/right maps. Separate them into .func.gii files.
echo "-> Separating CIFTI into left/right metric files..."
# Extract only the second map (index 1, which is k=2)
${WB_COMMAND} -cifti-merge ${OUT_PREFIX}_k2_only.dscalar.nii \
    -cifti ${anat_infomap} -column 2

# Then separate it normally (no COLUMN parameter needed)
${WB_COMMAND} -cifti-separate ${OUT_PREFIX}_k2_only.dscalar.nii COLUMN \
    -metric CORTEX_LEFT "${LEFT_METRIC}" \
    -metric CORTEX_RIGHT "${RIGHT_METRIC}"
echo "Left metric:  ${LEFT_METRIC}"
echo "Right metric: ${RIGHT_METRIC}"

# --------------------- STEP 2: (Optional) Create a new dscalar that uses our metrics as maps ----------
# This is useful if your template.scene references a dscalar file. We will create a minimal dscalar
# using the original as a template (keeps brain-model mappings identical).
echo "-> Recreating a dscalar that contains the two cortex metric maps (for templating)..."
# Use the original as a template to ensure consistent brain-model mapping
${WB_COMMAND} -cifti-create-dense-scalar "${OUT_CIFTI}" \
    -left-metric "${LEFT_METRIC}" \
    -right-metric "${RIGHT_METRIC}"

echo "Created dscalar: ${OUT_CIFTI}"

# --------------------- STEP 3: Create subject-specific scene ---------------------
echo "-> Creating subject-specific scene from template..."

SUBJ_SCENE="${OUT_PREFIX}_scene.scene"

# Copy template
cp "${TEMPLATE_SCENE}" "${SUBJ_SCENE}"

# Replace the filenames to match what the script generates
sed -i "s|sub-04_k2_left.func.gii|sub-${sub}_${APPROACH}.L.func.gii|g" "${SUBJ_SCENE}"
sed -i "s|sub-04_k2_right.func.gii|sub-${sub}_${APPROACH}.R.func.gii|g" "${SUBJ_SCENE}"
sed -i "s|sub-04|sub-${sub}|g" "${SUBJ_SCENE}"

# Replace ALL occurrences of sub-04 with current subject
# This catches the .func.gii files AND any dscalar references
sed -i "s|sub-04|sub-${sub}|g" "${SUBJ_SCENE}"

# Verify the replacement worked
echo "Checking scene file references..."
if grep -q "sub-04" "${SUBJ_SCENE}"; then
    echo "⚠ WARNING: Scene still contains sub-04 references!"
    grep "sub-04" "${SUBJ_SCENE}" | head -3
else
    echo "✓ Scene updated to reference sub-${sub}"
fi

# --------------------- STEP 4: Automated rendering ---------------------
echo "-> Rendering scene to PNG..."
"${WB_COMMAND}" -show-scene \
    "${SUBJ_SCENE}" \
    1 \
    "${OUT_IMAGE}" \
    1200 800

echo "✓ Created image: ${OUT_IMAGE}"
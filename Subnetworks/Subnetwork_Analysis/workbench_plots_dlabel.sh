#!/usr/bin/env bash
set -euo pipefail

# --------------------- USER SETTINGS ---------------------
sub="$1"

APPROACH="kmeans"   # Options: "infomap", "kmeans"
SURFACE="inflated"  # Options: "inflated", "very_inflated"

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

# Define both file paths
anat_infomap="${APPROACH_DIR}/sub-${sub}/${sub}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii"
anat_kmeans="${APPROACH_DIR}/sub-${sub}/sub-${sub}_kmeans_on_vertices_relabeled.dlabel.nii"

# Choose the correct input file based on APPROACH
if [ "${APPROACH}" = "kmeans" ]; then
    INPUT_FILE="${anat_kmeans}"
else
    INPUT_FILE="${anat_infomap}"
fi

# --------------------- OUTPUT FILES ---------------------
mkdir -p "${OUT_DIR}"
PNG_DIR="${SUBNETWORK_DIR}/brain_plots_k2_png"
mkdir -p "${PNG_DIR}"
OUT_PREFIX="${OUT_DIR}/sub-${sub}_${APPROACH}"
LEFT_METRIC="${OUT_PREFIX}.L.label.gii"
RIGHT_METRIC="${OUT_PREFIX}.R.label.gii"

# combined CIFTI (optionally rebuilt to match a template)
OUT_CIFTI="${OUT_PREFIX}.dlabel.nii"

# Template 
TEMPLATE_SCENE="${OUT_DIR}/template_dlabel.scene"   # "template_dlabel.scene" or template.scene
OUT_IMAGE="${PNG_DIR}/sub-${sub}_${APPROACH}.png"

# --------------------- CHECKS ---------------------
if [ -z "${WB_COMMAND}" ]; then
  echo "ERROR: wb_command not found on PATH. Please install Connectome Workbench and add wb_command to PATH."
  exit 2
fi

if [ ! -f "${INPUT_FILE}" ]; then
  echo "ERROR: input dlabel.nii not found: ${INPUT_FILE}"
  exit 3
fi

if [ ! -f "${SURF_L}" ] || [ ! -f "${SURF_R}" ]; then
  echo "ERROR: left/right surfaces not found."
  echo "  SURF_L=${SURF_L}"
  echo "  SURF_R=${SURF_R}"
  exit 4
fi

# --------------------- STEP 1: Extract left & right cortex LABEL files ---------------------
echo "-> Separating CIFTI into left/right label files..."

# Since the file only contains k=2 already, separate it directly:
${WB_COMMAND} -cifti-separate ${INPUT_FILE} COLUMN \
    -label CORTEX_LEFT "${LEFT_METRIC}" \
    -label CORTEX_RIGHT "${RIGHT_METRIC}"
    
echo "Left label:  ${LEFT_METRIC}"
echo "Right label: ${RIGHT_METRIC}"

# --------------------- STEP 2: Recreate dlabel from separated labels ----------
echo "-> Recreating a dlabel that contains the two cortex label maps (for templating)..."
${WB_COMMAND} -cifti-create-label "${OUT_CIFTI}" \
    -left-label "${LEFT_METRIC}" \
    -roi-left "${ROI_L}" \
    -right-label "${RIGHT_METRIC}" \
    -roi-right "${ROI_R}"

echo "Created dlabel: ${OUT_CIFTI}"

# --------------------- STEP 3: Create subject-specific scene ---------------------
echo "-> Creating subject-specific scene from template..."

SUBJ_SCENE="${OUT_PREFIX}_scene.scene"

# Copy template for all subjects
cp "${TEMPLATE_SCENE}" "${SUBJ_SCENE}"

# For non-07 subjects, update subject ID
if [ "${sub}" != "07" ]; then
    # Replace subject ID (sub-07 from template)
    sed -i "s|sub-07|sub-${sub}|g" "${SUBJ_SCENE}"
    
    # Update the filename pattern to include current subject ID
    sed -i "s|07_FPN_infomap|${sub}_FPN_infomap|g" "${SUBJ_SCENE}"
fi

# Update approach-specific paths for ALL subjects (including 07)
if [ "${APPROACH}" = "kmeans" ]; then
    # Replace approach in path
    sed -i "s|/infomap/|/${APPROACH}/|g" "${SUBJ_SCENE}"
    
    # Update filename pattern for kmeans
    sed -i "s|${sub}_FPN_infomap_communities_kmeans_relabeled|sub-${sub}_kmeans_on_vertices_relabeled|g" "${SUBJ_SCENE}"
    sed -i "s|07_FPN_infomap_communities_kmeans_relabeled|sub-${sub}_kmeans_on_vertices_relabeled|g" "${SUBJ_SCENE}"
fi

# Verify the replacement worked
echo "Checking scene file references..."
if [ "${sub}" != "07" ] && grep -q "sub-07" "${SUBJ_SCENE}"; then
    echo "⚠ WARNING: Scene still contains sub-07 references!"
    grep "sub-07" "${SUBJ_SCENE}" | head -3
else
    echo "✓ Scene updated to reference sub-${sub} with approach ${APPROACH}"
fi

# --------------------- STEP 4: Automated rendering ---------------------
echo "-> Rendering scene to PNG..."
"${WB_COMMAND}" -show-scene \
    "${SUBJ_SCENE}" \
    1 \
    "${OUT_IMAGE}" \
    1200 800

echo "✓ Created image: ${OUT_IMAGE}"
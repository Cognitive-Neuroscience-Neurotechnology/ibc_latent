#!/usr/bin/env bash
# Batch plot k=2 subnetworks for a subject (CORTEX ONLY)

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <subject_id_or_sub-XX>"
  exit 1
fi

raw_sub="$1"
if [[ "${raw_sub}" =~ ^sub-([0-9]+)$ ]]; then
  subj="$(printf "%02d" "${BASH_REMATCH[1]}")"
  sub="sub-${subj}"
elif [[ "${raw_sub}" =~ ^[0-9]+$ ]]; then
  subj="$(printf "%02d" "${raw_sub}")"
  sub="sub-${subj}"
else
  echo "Invalid subject: ${raw_sub}" >&2
  exit 1
fi

WORKING_DIR="/ptmp/hmueller2/Downloads"
SUBNETWORK_DIR="${WORKING_DIR}/subnetworks"
INFOMAP_DIR="${SUBNETWORK_DIR}/infomap"
MASKS_DIR="${WORKING_DIR}/fsLR_masks"
OUT_DIR="${SUBNETWORK_DIR}/brain_plots_k2_wb"
mkdir -p "${OUT_DIR}"

SURF_L="${MASKS_DIR}/fs_LR.32k.L.very_inflated.surf.gii"
SURF_R="${MASKS_DIR}/fs_LR.32k.R.very_inflated.surf.gii"
ROI_L="${MASKS_DIR}/L.atlasroi.32k_fs_LR.shape.gii"
ROI_R="${MASKS_DIR}/R.atlasroi.32k_fs_LR.shape.gii"

INPUT_CIFTI="${INFOMAP_DIR}/${sub}/${subj}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii"

# --------------------

if [[ ! -f "${INPUT_CIFTI}" ]]; then
  echo "Missing input CIFTI: ${INPUT_CIFTI}" >&2
  exit 2
fi

command -v wb_command >/dev/null 2>&1 || { echo "wb_command not found in PATH" >&2; exit 3; }

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# Get map count and ensure at least 2
n_maps="$(wb_command -file-information "${INPUT_CIFTI}" | awk '/Maps:/ {print $NF; exit}')"
if ! [[ "${n_maps}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse map count; defaulting to 1" >&2
  n_maps=1
fi
echo "Detected map count: ${n_maps}"
if [[ "${n_maps}" -lt 2 ]]; then
  echo "Need at least 2 maps to extract k=2 (found ${n_maps})" >&2
  exit 4
fi

# Extract map 2 (k=2) to its own dscalar
k2_scalar="${tmpdir}/k2.dscalar.nii"
wb_command -cifti-merge "${k2_scalar}" -cifti "${INPUT_CIFTI}" -column 2

# Separate cortex from extracted single-map dscalar
wb_command -cifti-separate "${k2_scalar}" COLUMN \
  -metric CORTEX_LEFT  "${tmpdir}/left_raw.func.gii" \
  -metric CORTEX_RIGHT "${tmpdir}/right_raw.func.gii"

wb_command -metric-mask "${tmpdir}/left_raw.func.gii"  "${ROI_L}" "${tmpdir}/left.func.gii"
wb_command -metric-mask "${tmpdir}/right_raw.func.gii" "${ROI_R}" "${tmpdir}/right.func.gii"

wb_command -metric-palette "${tmpdir}/left.func.gii"  MODE_AUTO_SCALE -palette-name Classic -disp-neg false -disp-pos true
wb_command -metric-palette "${tmpdir}/right.func.gii" MODE_AUTO_SCALE -palette-name Classic -disp-neg false -disp-pos true

# Create a cortex-only dscalar (optional output for easier viewing in wb_view)
cortex_scalar="${OUT_DIR}/${sub}_k2_cortex.dscalar.nii"
wb_command -cifti-create-dense-scalar "${cortex_scalar}" \
  -left-metric "${tmpdir}/left.func.gii" \
  -right-metric "${tmpdir}/right.func.gii"

# Save metrics for manual inspection
cp "${tmpdir}/left.func.gii"  "${OUT_DIR}/${sub}_k2_left.func.gii"
cp "${tmpdir}/right.func.gii" "${OUT_DIR}/${sub}_k2_right.func.gii"

echo "Saved metrics:"
echo "  ${OUT_DIR}/${sub}_k2_left.func.gii"
echo "  ${OUT_DIR}/${sub}_k2_right.func.gii"
echo "Combined cortex scalar:"
echo "  ${cortex_scalar}"
echo "Open these in wb_view manually (segfault avoided by skipping scene capture)."

# Optional (unstable) scene capture only if SCENE_CAPTURE=1
if [[ "${SCENE_CAPTURE:-0}" == "1" ]]; then
  echo "Attempting scene capture (experimental)..."
  scene_file="${tmpdir}/lr.scene"
  cat > "${scene_file}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<SceneFile Version="3">
  <Scene Name="Left" NumberOfBrowserWindows="1">
    <BrowserWindow NumberOfTabs="1">
      <Tab Name="Layers" Selected="true">
        <Brain Name="Surface" Selected="true">
          <Surface Open="true" FileName="${SURF_L}"/>
          <Overlay Name="Primary Overlay" DataType="Metric" FileName="${OUT_DIR}/${sub}_k2_left.func.gii" SelectedMapIndex="1"/>
        </Brain>
      </Tab>
    </BrowserWindow>
  </Scene>
  <Scene Name="Right" NumberOfBrowserWindows="1">
    <BrowserWindow NumberOfTabs="1">
      <Tab Name="Layers" Selected="true">
        <Brain Name="Surface" Selected="true">
          <Surface Open="true" FileName="${SURF_R}"/>
          <Overlay Name="Primary Overlay" DataType="Metric" FileName="${OUT_DIR}/${sub}_k2_right.func.gii" SelectedMapIndex="1"/>
        </Brain>
      </Tab>
    </BrowserWindow>
  </Scene>
</SceneFile>
EOF
  left_png="${OUT_DIR}/${sub}_k2_left.png"
  right_png="${OUT_DIR}/${sub}_k2_right.png"
  if wb_command -scene-capture-image "${scene_file}" 1 "${left_png}"  -size-width-height 800 800 -renderer OSMesa \
     && wb_command -scene-capture-image "${scene_file}" 2 "${right_png}" -size-width-height 800 800 -renderer OSMesa; then
    echo "Captured images (experimental):"
    echo "  ${left_png}"
    echo "  ${right_png}"
  else
    echo "Scene capture failed or segfaulted; images not produced." >&2
  fi
fi
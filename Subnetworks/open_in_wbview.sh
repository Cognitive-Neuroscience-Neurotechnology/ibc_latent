#!/bin/bash
# chmod +x /Users/hannahmuller/Downloads/open_in_wbview.sh
# Usage: /Users/hannahmuller/Downloads/open_in_wbview.sh 04

SUB="$1"

if [[ -z "$SUB" ]]; then
	echo "Usage: $0 <SUBJECT_ID>   e.g., $0 04"
	exit 1
fi

# Ensure globs that match nothing expand to nothing (not the literal pattern)
shopt -s nullglob

# —— Base directories ——
NETWORKS_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/individual_networks"
SUBNETWORKS_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/subnetworks"
FMRIPREP_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/fmriprep_out"
CONTRAST_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/contrast_maps_fsLR"
MNI_DIR="/Users/hannahmuller/nyx_mount_ptmp/Downloads/ibc_contrast_maps/resulting_smooth_maps_surface"

SESSION="38"
TASK="VisualSearch"

# Add Workbench to PATH
export PATH="/Users/hannahmuller/workbench/bin_macosxub:$PATH"

# ==============================
# —— ALWAYS USE: Surfaces ——
# ==============================
SURF_L="${FMRIPREP_DIR}/sub-${SUB}/anat/sub-${SUB}_hemi-L_inflated.32k_fs_LR.surf.gii"
SURF_R="${FMRIPREP_DIR}/sub-${SUB}/anat/sub-${SUB}_hemi-R_inflated.32k_fs_LR.surf.gii"

# =========================================
# —— 1. Task time series (task GLM inputs) ——
# =========================================
UNSCRUBBED="${FMRIPREP_DIR}/sub-${SUB}/ses-${SESSION}/postfmriprep/GLM/sub-${SUB}_ses-${SESSION}_task-${TASK}_dir-ap_cleaned_noscrub.dtseries.nii"
SCRUBBED="${FMRIPREP_DIR}/sub-${SUB}/ses-${SESSION}/postfmriprep_scrubbed/GLM/sub-${SUB}_ses-${SESSION}_task-${TASK}_dir-ap_cleaned.dtseries.nii"

# ===========================================
# —— 2. MNI timeseries ZMaps (surface GIFTI) ——
# ===========================================
MNILEFT=(  "${MNI_DIR}/sub-${SUB}/ses-${SESSION}/sub-${SUB}_ses-${SESSION}_task-${TASK}_dir-ffx_space-individual_hemi-lh_ZMap-delay_wm.gii" )
MNIRIGHT=( "${MNI_DIR}/sub-${SUB}/ses-${SESSION}/sub-${SUB}_ses-${SESSION}_task-${TASK}_dir-ffx_space-individual_hemi-rh_ZMap-delay_wm.gii" )

# ===================================
# —— 3. Contrast maps (CIFTI dscalar) ——
# ===================================
# Include specific file and allow future expansion via globs if present
CONTRASTMAPS=(
	"${CONTRAST_DIR}/sub-${SUB}/ses-${SESSION}/res_task-${TASK}_space-fsLR_run-01_dir-pa/z_score_maps/delay_wm.dscalar.nii"
	"${CONTRAST_DIR}/sub-${SUB}/ses-${SESSION}/res_task-${TASK}_space-fsLR_dir-pa/z_score_maps/"*.dscalar.nii
	"${CONTRAST_DIR}/sub-${SUB}/ses-${SESSION}/res_task-${TASK}_space-fsLR_run-01_dir-pa/z_score_maps/"*.dscalar.nii
)

# ===========================================================
# —— 4. Concatenated resting-state time series (Infomap base) ——
# ===========================================================
# Bring in all variants mentioned on the page:
# - cleaned (with coupling correction)
# - cleaned_but_coupled (without coupling correction)
# - smoothed_* (0.85 / 1.7 / 2.55)
CONCAT_TS=(
	#"${NETWORKS_DIR}/sub-${SUB}/resting_state/sub-${SUB}_all-tasks_concatenated_cleaned_fsLR.dtseries.nii"
	#"${NETWORKS_DIR}/sub-${SUB}/resting_state/sub-${SUB}_all-tasks_concatenated_cleaned_but_coupled_fsLR.dtseries.nii"
	"${NETWORKS_DIR}/sub-${SUB}/resting_state/sub-${SUB}_all-tasks_concatenated_cleaned_smoothed_2.55_fsLR.dtseries.nii"
)

# ===================================================
# —— 5. Baseline communities and filtered communities ——
# ===================================================
COMMUNITIES=( "${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities.dtseries.nii" )
SPATIAL_FILTER_COMMUNITIES=( "${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii" )
SPATIAL_FILTER_COMMUNITIES_ALT=( "${NETWORKS_DIR}/sub-${SUB}/resting_state_0.85/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii" )

# ==========================================
# —— 6. Algorithmic labeling and derivatives ——
# ==========================================
# Dense label of labeled communities
LABELED_COMMUNITIES="${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii"
LABELED_COMMUNITIES_ALT="${NETWORKS_DIR}/sub-${SUB}/resting_state_0.85/Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii"

# Optional: “InfoMapCommunitiesf” labeled output (if produced)
INFOMAP_COMMUNITIES=( "${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+AlgorithmicLabeling_InfoMapCommunities.dlabel.nii" )

# Borders for L/R (if present)
BORDERS=(
	"${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+AlgorithmicLabeling.L.border"
	"${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+AlgorithmicLabeling.R.border"
)

# FC derivatives referenced on page (if present)
FC_WHOLEBRAIN=( "${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+AlgorithmicLabeling_FC_WholeBrain.dtseries.nii" )
FC_BTWN_COMMUNITIES=( "${NETWORKS_DIR}/sub-${SUB}/resting_state/Bipartite_PhysicalCommunities+AlgorithmicLabeling_FC_btwn_InfoMapCommunities.dtseries.nii" )

# ====================================
# —— 7. Individual nets concatenations ——
# ====================================
INDIVIDNETS="${NETWORKS_DIR}/sub-${SUB}/resting_state/sub-${SUB}_individual_nets_concat.ptseries.nii"

# =====================================
# —— 8. FPN: Mask & Subnetwork labeling ——
# =====================================
FPN_ROI="${NETWORKS_DIR}/sub-${SUB}/resting_state/Frontoparietal_roi.dscalar.nii"
FPN_COMMUNITIES="${SUBNETWORKS_DIR}/infomap/sub-${SUB}/FPN_communities.dscalar.nii"

# ===========================================
# —— 9. K-means inside FPN (verification sets) ——
# ===========================================
#COM_KMEANSDTSERIES="${SUBNETWORKS_DIR}/infomap/sub-${SUB}/${SUB}_FPN_infomap_communities_kmeans.dtseries.nii"
COM_KMEANSLABEL="${SUBNETWORKS_DIR}/infomap/sub-${SUB}/${SUB}_FPN_infomap_communities_kmeans.dlabel.nii"
COM_KMEANSDSCALAR="${SUBNETWORKS_DIR}/infomap/sub-${SUB}/${SUB}_FPN_infomap_communities_kmeans.dscalar.nii"
VER_KMEANS="${SUBNETWORKS_DIR}/kmeans/sub-${SUB}/sub-${SUB}_kmeans_on_vertices.dtseries.nii"

# ======================
# —— Launch wb_view ——
# ======================
args=(
  "$SURF_L" "$SURF_R"

  # --- 1. Task time series (fmriprep output)
  #"$SCRUBBED" "$UNSCRUBBED"

  # --- 2. MNI surface ZMaps
  #"${MNILEFT[@]}"
  #"${MNIRIGHT[@]}"

  # --- 3. Contrast maps
 #"${CONTRASTMAPS[@]}"

  # --- 4. Concatenated resting-state time series (Infomap base)
  #"${CONCAT_TS[@]}"

  # --- 5. Infomap whole brain communities
  #"${COMMUNITIES[@]}"
  #"${SPATIAL_FILTER_COMMUNITIES[@]}"
  #"${SPATIAL_FILTER_COMMUNITIES_ALT[@]}"

  # --- 6. Infomap networks: Algorithmic labeling and derivatives
  #"$LABELED_COMMUNITIES"
  #"$LABELED_COMMUNITIES_ALT"
  #"${INFOMAP_COMMUNITIES[@]}"
  #"${BORDERS[@]}"
  #"${FC_WHOLEBRAIN[@]}"
  #"${FC_BTWN_COMMUNITIES[@]}"

  # --- 7. Time series of individual networks (ptseries of RS)
  #"$INDIVIDNETS"

  # --- 8. FPN: Mask & communities
  #"$FPN_ROI"
  #"$FPN_COMMUNITIES"

  # --- 9. FPN: Subnetworks via K-means
  #"$COM_KMEANSDTSERIES"
  #"#$COM_KMEANSLABEL"
  "$COM_KMEANSDSCALAR"
  #"$VER_KMEANSFC"
  #"$VER_KMEANS" 
)


echo "Launching wb_view..."
wb_view "${args[@]}"
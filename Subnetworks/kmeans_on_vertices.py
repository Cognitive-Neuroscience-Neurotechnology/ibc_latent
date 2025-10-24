'''
(2)
kmeans clustering on all vertices of the FPN based on their connectivity profiles within the FPN.
'''

"""
Identify subnetworks using k-means like DiNicola et al., 2022 (https://doi.org/10.1152/jn.00211.2022)

1. z-score and concatenate timeseries
2. input timeseries into k-means (MATLAB) using default parameters
    vary k=10-20, chose k=6 
    (why k=6 is not mentioned in the methods, but I haven't read the whole paper so might be explained somewhere else)
3. identify networks based on "referential features" (cite Braga & Buckner, 2017) 

"""

import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import argparse
import os
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# TO USE THE ONE THAT IS SMOOTHED??
## TO USE FC OR RAW TIMESERIES??
### To use concatenated tasks or just resting state?


parser = argparse.ArgumentParser(description='a script to run k-means on the FPN')
parser.add_argument('--subject', help='do not include sub- prefix', default='06')
#parser.add_argument('--seq', help='sequenceName, e.g., task-rest_acq-lowresmb.')
#parser.add_argument('--half', help='Half of dataset to run across (odd or even).')
args = parser.parse_args()

#sequenceName = args.seq
subject = args.subject
#half = args.half

"""
00. Set up directories
"""
print("----- 00. Set up directories -----")

working_dir = '/ptmp/hmueller2/Downloads'
timeseries_dir = os.path.join(working_dir, "fmriprep_out", f'sub-{subject}')
network_dir = os.path.join(working_dir, "individual_networks", f'sub-{subject}')
output_dir = os.path.join(working_dir, "subnetworks", "kmeans", f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

session_dirs = RR.get_sessions_dirs(timeseries_dir)
#half_sessions = [] # see if i need this
#files_subjects=[]

"""
01. Load data, concatenate & z-score
"""
print("----- 01. Load / concatenate data -----")

# Option A: Concatenate all sessions (with split half option)
'''
# Concatenate runs
for session in session_dirs:
    print(f'--- Session: {session} ---')

    current_dir = os.path.join(timeseries_dir, session, 'postfmriprep', 'GLM')
    print("Current directory:", current_dir)

    pattern = re.compile(rf'sub-{subject}_{session}_task-.*_dir-.*_cleaned_noscrub\.dtseries\.nii')
    run_list = RR.whicth_runs(current_dir, pattern)

    # Construct filename
    files_per_session = []
    for run in run_list:
        # run is now the full filename (since run number is not in the filename)
        filenamebase = f'sub-{subject}_{session}_task-{sequenceName}_dir-{run}_cleaned_noscrub.dtseries.nii'
        file = os.path.join(current_dir, filenamebase)
        files_per_session.append(file)
    
    # Concatenate
    print("Concatenating runs...")
    output_filename = os.path.join(current_dir, f'sub-{subject}_{session}_task-{sequenceName}_concat.LR.32k.dtseries.nii')
    files_subjects.append(output_filename)
    # If you still want split half, need to implement that in concat_WB, so outcomment the next line
    # RR.concat_WB(files_per_session, output_filename, return_data=False)

    # Extract even or odd for split-half
    session_number = int(session.split('ses-')[1])  
    if half == 'odd' and session_number % 2 != 0:
        half_sessions.append(session_number)
    elif half == 'even' and session_number % 2 == 0:
        half_sessions.append(session_number)

print("Half sessions:", half_sessions)

print("Concatenating sessions...")
files_to_concat=RR.collect_files_to_concat(files_subjects, half_sessions) # need to implement loop with halves
all_data_concat=RR.concat_files(files_to_concat)
# filename=os.path.join(network_dir, f'{half}_half', f'sub-{subject}'+ f'_{half}' + f'_{sequenceName}_concatenated_smoothed0.85_masked_32k_fsLR.dtseries.nii')
# all_data_concat = RR.load_data(filename)
n_vertices=all_data_concat.shape[1]
'''

# ---- INPUT DATA ----

# Option B: Load pre-concatenated data
full_concat = os.path.join(
    working_dir, "individual_networks", f"sub-{subject}", "resting_state",
    f"sub-{subject}_all-tasks_concatenated_cleaned_fsLR.dtseries.nii")
print(f"Loading prebuilt concatenated dtseries:\n  {full_concat}")
all_data_concat = RR.load_data(full_concat)  # shape: (time, grayordinates)
n_vertices = all_data_concat.shape[1]

# 1) 91k -> 2) restrict to 64k cortex
dtseries_cortex = all_data_concat[:, :64984]
print("Restricted dtseries to cortex:", dtseries_cortex.shape)

# 3) Load FPN ROI and restrict to 64k, then extract FPN vertices
FPN_file = os.path.join(network_dir, 'resting_state', 'Frontoparietal_roi.dscalar.nii')
fpn_mask = RR.load_data(FPN_file)  # expects shape (1, 91282) or (91282,)
fpn_mask = np.squeeze(fpn_mask)
if fpn_mask.ndim != 1:
    raise RuntimeError(f"Unexpected FPN mask ndim={fpn_mask.ndim}")
if fpn_mask.shape[0] == 91282:
    print("FPN mask has 91k vertices, restricting to cortex...")
    fpn_mask = fpn_mask[:64984]  # cortex only
fpn_mask_bool = fpn_mask.astype(bool)
n_fpn = int(fpn_mask_bool.sum())
print(f"FPN vertices (cortex-only): {n_fpn}")

# Indices of FPN vertices in the template (restrict to cortex slice in the 91k template)
fpn_mask_full = np.squeeze(RR.load_data(FPN_file))
if fpn_mask_full.ndim != 1:
    raise RuntimeError(f"Unexpected FPN mask ndim in file={fpn_mask_full.ndim}")
fpn_indices = np.where(fpn_mask_full[:64984].astype(bool))[0]  # cortical indices only
print(f"FPN cortical indices in 91k template: {fpn_indices.shape[0]}")
assert fpn_indices.shape[0] == n_fpn, f"ids length {fpn_indices.shape[0]} != N_fpn {n_fpn}"

# Extract FPN timeseries (T x N_fpn)
FPN_data = dtseries_cortex[:, fpn_mask_bool]

print("Z-scoring FPN timeseries (over time)...")
Z_tv = RR.z_score_np(FPN_data)  # shape: (T, N_fpn)
print("Z-scored FPN shape:", Z_tv.shape)

# DiNicola method: cluster on raw timecourses
X = Z_tv.T  # shape: (N_fpn, T)

n_clusters = list(range(2, 11))  # 2..10 inclusive
dtseries_template = full_concat
half_label = 'all'
output_filename = os.path.join(output_dir, f"sub-{subject}_{half_label}_FPN_kmeans_RAWTIME.dtseries.nii")
print(f"dtseries template: {dtseries_template}")
print(f"output dtseries:   {output_filename}")

cluster_results_raw, inertia_raw, silhouette_scores_raw, kmeans_list_raw = RR.kmeans_standard(
    n_clusters,
    X,                       # cluster on raw timecourses
    save_to_file=True,
    remap_to_verts=True,
    filename=output_filename,
    mask_file=FPN_file,
    dtseries_template=dtseries_template,
    ids=fpn_indices
)

# Sanity check for RAWTIME
RR.check_mask_template_alignment(X.shape[0], FPN_file, fpn_indices, dtseries_template)

"""
03. Kmeans clustering (on Functional Connectivity of all vertices within FPN with each other)
"""
print("----- 03. Kmeans clustering (on Functional Connectivity of all FPN vertices) -----")

# Build within-FPN FC features via PCA basis on Z_tv (time x vertices)
m = 20  # number of components for low-dim FC representation
pca = PCA(n_components=min(m, Z_tv.shape[0]-1, Z_tv.shape[1]), svd_solver='randomized', random_state=0)
comp_ts = pca.fit_transform(Z_tv)  # (T, m_eff)
# z-score component timecourses over time
comp_ts = (comp_ts - np.mean(comp_ts, axis=0, keepdims=True))
comp_std = np.std(comp_ts, axis=0, ddof=1, keepdims=True); comp_std[comp_std == 0] = 1e-8
comp_ts = comp_ts / comp_std

# Correlate each vertex timeseries with each component timecourse → features (N_fpn × m_eff)
Zc = Z_tv - np.mean(Z_tv, axis=0, keepdims=True)
Zstd = np.std(Zc, axis=0, ddof=1, keepdims=True); Zstd[Zstd == 0] = 1e-8
Zc = Zc / Zstd
Tlen = Zc.shape[0]
features = (Zc.T @ comp_ts) / (Tlen - 1)
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize features for k-means
features = StandardScaler().fit_transform(features)

# 5) k-means on FC features (k=2..10), remap labels back to vertices
n_clusters = list(range(2, 11))  # 2..10 inclusive
dtseries_template = full_concat  # OK to write into 91k template; ROI is cortical
half_label = 'all'
output_filename = os.path.join(output_dir, f"sub-{subject}_{half_label}_FPN_kmeans_FCwithin.dtseries.nii")
print(f"dtseries template: {dtseries_template}")
print(f"output dtseries:   {output_filename}")

cluster_results_fc, inertia_fc, silhouette_scores_fc, kmeans_list_fc = RR.kmeans_standard(
    n_clusters,
    features,                 # cluster on FC features
    save_to_file=True,
    remap_to_verts=True,      # remap labels back to FPN vertices
    filename=output_filename,
    mask_file=FPN_file,       # ROI defines which vertices receive labels
    dtseries_template=dtseries_template,
    ids=fpn_indices           # IMPORTANT: place labels at correct vertices
)

# Sanity check for FCwithin
RR.check_mask_template_alignment(features.shape[0], FPN_file, fpn_indices, dtseries_template)

"""
04. Plots and metrics
"""
# RAWTIME plots
elbow_path_raw = os.path.join(output_dir, f"sub-{subject}_{half_label}_RAWTIME_elbow.png")
RR.elbow_plot(n_clusters, inertia_raw, elbow_path_raw)

sil_path_raw = os.path.join(output_dir, f"sub-{subject}_{half_label}_RAWTIME_silhouette.png")
RR.silhouette_plot(silhouette_scores_raw, sil_path_raw)

print("\n=== RAWTIME metrics ===")
for i, k in enumerate(n_clusters):
    labels = cluster_results_raw[i, :]
    entropy = RR.compute_entropy(labels)
    bic = RR.compute_bic(kmeans_list_raw[k-1], X)  # use raw feature matrix X
    smallest_size = RR.smallest_cluster_size(labels)
    print(f"k={k}: Entropy={entropy:.4f}  BIC={bic:.2f}  Smallest={smallest_size}")

# FCwithin plots
elbow_path_fc = os.path.join(output_dir, f"sub-{subject}_{half_label}_FCwithin_elbow.png")
RR.elbow_plot(n_clusters, inertia_fc, elbow_path_fc)

sil_path_fc = os.path.join(output_dir, f"sub-{subject}_{half_label}_FCwithin_silhouette.png")
RR.silhouette_plot(silhouette_scores_fc, sil_path_fc)

print("\n=== FCwithin metrics ===")
for i, k in enumerate(n_clusters):
    labels = cluster_results_fc[i, :]
    entropy = RR.compute_entropy(labels)
    bic = RR.compute_bic(kmeans_list_fc[k-1], features)  # use FC feature matrix
    smallest_size = RR.smallest_cluster_size(labels)
    print(f"k={k}: Entropy={entropy:.4f}  BIC={bic:.2f}  Smallest={smallest_size}")
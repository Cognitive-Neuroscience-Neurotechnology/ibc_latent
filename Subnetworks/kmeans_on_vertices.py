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
import csv


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
# Option A: Concatenate all sessions (with split half option)
'''
print("----- 01. Concatenate data -----")

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
print("----- 01. Load concatenated data -----")

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

# DiNicola method: cluster on raw timecourses (RAWTIME → now just "kmeans on vertices")
X = Z_tv.T  # shape: (N_fpn, T)

n_clusters = list(range(2, 11))  # 2..10 inclusive
dtseries_template = full_concat

# NEW filenames (no split-half label)
output_filename = os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices.dtseries.nii")
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
03. Plots and metrics
"""
# RAWTIME plots (renamed to kmeans_on_vertices)
elbow_path_raw = os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices_elbow.png")
RR.elbow_plot(n_clusters, inertia_raw, elbow_path_raw)

sil_path_raw = os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices_silhouette.png")
RR.silhouette_plot(silhouette_scores_raw, sil_path_raw)

print("\n=== kmeans_on_vertices metrics ===")
# COLLECT metrics per k for CSV (kmeans on vertices)
entropy_list_raw, bic_list_raw, smallest_list_raw = [], [], []
k_values = list(n_clusters)
for i, k in enumerate(n_clusters):
    labels = cluster_results_raw[i, :]
    entropy = RR.compute_entropy(labels)
    # Robust BIC: handle missing model objects
    kmeans_model = kmeans_list_raw[i] if i < len(kmeans_list_raw) else None
    if kmeans_model is None:
        bic = np.nan
    else:
        try:
            bic = RR.compute_bic(kmeans_model, X)
        except Exception as e:
            print(f"[warn] BIC failed for k={k}: {e}")
            bic = np.nan
    smallest_size = RR.smallest_cluster_size(labels)
    print(f"k={k}: Entropy={entropy:.4f}  BIC={bic}  Smallest={smallest_size}")
    entropy_list_raw.append(float(entropy))
    bic_list_raw.append(float(bic) if np.isfinite(bic) else np.nan)
    smallest_list_raw.append(int(smallest_size))

# Helper to align sequences to k-values
def series_for_k(seq_or_map, ks):
    # Accept list/np.array (index by position) or dict (index by k)
    if isinstance(seq_or_map, dict):
        vals = [seq_or_map.get(k, np.nan) for k in ks]
    else:
        seq = list(seq_or_map)
        if len(seq) < len(ks):
            seq = seq + [np.nan] * (len(ks) - len(seq))
        elif len(seq) > len(ks):
            seq = seq[:len(ks)]
        vals = seq
    return [float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else '' for v in vals]

# Prepare inertia and silhouette series aligned to k
inertia_series_raw = series_for_k(inertia_raw, k_values)
silhouette_series_raw = series_for_k(silhouette_scores_raw, k_values)

# WRITE CSV (kmeans on vertices) - renamed per your request
csv_path_raw = os.path.join(output_dir, f"sub-{subject}_clustering_of_kmeans_metrics.csv")
with open(csv_path_raw, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['metric'] + [f'k={k}' for k in k_values]
    writer.writerow(header)
    writer.writerow(['Entropy'] + entropy_list_raw)
    writer.writerow(['BIC'] + [v if v != '' else '' for v in bic_list_raw])
    writer.writerow(['Smallest Cluster Size'] + smallest_list_raw)
    writer.writerow(['Inertia'] + inertia_series_raw)
    writer.writerow(['Silhouette Score'] + silhouette_series_raw)
print(f"Saved metrics CSV: {csv_path_raw}")

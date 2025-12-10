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
from sklearn.metrics import silhouette_score
# Try real spherecluster; fall back to sklearn KMeans on unit vectors (fast) if unavailable
# Try real spherecluster; add a shim for sklearn>=1.2 if needed, else fallback.
_SphereKMeans = None
_HAS_SPHERECLUSTER = False
try:
    from spherecluster import SphericalKMeans as _SphereKMeans
    _HAS_SPHERECLUSTER = True
    print("[info] Using spherecluster as-is.")
except Exception as e1:
    # Shim sklearn.cluster.k_means_ -> sklearn.cluster._kmeans for newer scikit-learn
    try:
        import sys, types
        import sklearn.cluster._kmeans as _sk_kmeans
        shim = types.ModuleType("sklearn.cluster.k_means_")
        # Map old names used by spherecluster to new ones
        # spherecluster imports one or more of these:
        shim._k_init = getattr(_sk_kmeans, "_k_init")
        shim._labels_inertia = getattr(_sk_kmeans, "_labels_inertia")
        # Provide legacy alias if spherecluster looks for _init_centroids
        def _init_centroids(X, n_clusters, init, random_state, x_squared_norms=None, init_size=None):
            return _sk_kmeans._k_init(X, n_clusters, init, random_state, x_squared_norms=x_squared_norms)
        shim._init_centroids = _init_centroids
        # Optional tolerance helper (may not exist in newer sklearn)
        if hasattr(_sk_kmeans, "_tolerance"):
            shim._tolerance = _sk_kmeans._tolerance
        sys.modules["sklearn.cluster.k_means_"] = shim

        from spherecluster import SphericalKMeans as _SphereKMeans
        _HAS_SPHERECLUSTER = True
        print("[info] Enabled spherecluster via sklearn compatibility shim.")
    except Exception as e2:
        from sklearn.cluster import KMeans as _KMeans
        _SphereKMeans = None
        _HAS_SPHERECLUSTER = False
        print(f"[warn] spherecluster not available; fallback to sklearn KMeans on L2-normalized data. Reason: {e1} / {e2}")


parser = argparse.ArgumentParser(description='a script to run k-means on the FPN')
parser.add_argument('--subject', help='do not include sub- prefix', default='06')
args = parser.parse_args()

subject = args.subject

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

"""
01. Load data, concatenate & z-score
"""
# ---- INPUT DATA ----

# Option B: Load pre-concatenated data
print("----- 01. Load concatenated data -----")

full_concat = os.path.join(
    working_dir, "individual_networks", f"sub-{subject}", "resting_state",
    f"sub-{subject}_all-tasks_concatenated_cleaned_smoothed_2.55_fsLR.dtseries.nii")
print(f"Loading prebuilt concatenated dtseries:\n  {full_concat}")
all_data_concat = RR.load_data(full_concat)  # shape: (time, grayordinates)
n_vertices = all_data_concat.shape[1]

# 1) 91k -> 2) restrict to 64k cortex
dtseries_cortex = all_data_concat[:, :64984]
print("Restricted dtseries to cortex:", dtseries_cortex.shape)

# 2) Load FPN ROI and restrict to 64k, then extract FPN vertices
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

# Z-score FPN timeseries over time
print("Z-scoring FPN timeseries (over time)...")
Z_tv = RR.z_score_np(FPN_data)  # shape: (T, N_fpn)
print("Z-scored FPN shape:", Z_tv.shape)

# DiNicola method: cluster on raw timecourses
X = Z_tv.T  # shape: (N_fpn, T)

# L2-normalize each vertex timecourse (row) → spherical k-means (cosine-equivalent)
row_norms = np.linalg.norm(X, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
X = X / row_norms
print("Applied L2 normalization to vertex timecourses (spherical k-means).")


"""
02. Run k-means clustering on vertices within FPN
"""
n_clusters = list(range(2, 11))  # 2..10 inclusive
dtseries_template = full_concat

output_filename = os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices.dtseries.nii")
print(f"dtseries template: {dtseries_template}")
print(f"output dtseries:   {output_filename}")

# --- True spherical k-means (cosine objective) or fast fallback ---
cluster_results_raw = np.zeros((len(n_clusters), X.shape[0]), dtype=int)
inertia_raw, silhouette_scores_raw, kmeans_list_raw = [], [], []

for i, k in enumerate(n_clusters):
    if _HAS_SPHERECLUSTER:
        skm = _SphereKMeans(
            n_clusters=k,
            n_init=20,
            max_iter=300,
            init="k-means++",
            random_state=0,
            verbose=False,
        )
        labels_zero_based = skm.fit_predict(X)
        model = skm
        inertia_val = float(skm.inertia_)
    else:
        km = _KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=0, verbose=0)
        labels_zero_based = km.fit_predict(X)
        model = km
        inertia_val = float(km.inertia_)

    labels_one_based = labels_zero_based + 1
    cluster_results_raw[i, :] = labels_one_based
    kmeans_list_raw.append(model)
    inertia_raw.append(inertia_val)

    try:
        silhouette_scores_raw.append(float(silhouette_score(X, labels_zero_based, metric="cosine")))
    except Exception:
        silhouette_scores_raw.append(np.nan)

# Save as dtseries using RR.nib_save
# Create full brain array (n_clusters × 91282 grayordinates)
all_k_full = np.zeros((len(n_clusters), 91282), dtype=np.float32)
for i, k in enumerate(n_clusters):
    # Place 1-based labels at FPN cortex indices
    all_k_full[i, fpn_indices] = cluster_results_raw[i, :]

try:
    RR.nib_save(output_filename, all_k_full, dtseries_template)
    print(f"Saved dtseries: {output_filename}")
except Exception as e:
    print(f"[warn] Failed to save dtseries: {e}")

# Sanity check
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
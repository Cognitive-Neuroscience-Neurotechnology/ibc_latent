'''
(2)
kmeans clustering on all vertices of the FPN based on their connectivity profiles within the FPN.
'''

import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import argparse
import os
import numpy as np
import csv

parser = argparse.ArgumentParser(description='k-means on FPN vertices')
parser.add_argument('--subject', help='do not include sub- prefix', default='06')
args = parser.parse_args()

subject = args.subject

# Setup directories
working_dir = '/ptmp/hmueller2/Downloads'
network_dir = os.path.join(working_dir, "individual_networks", f'sub-{subject}')
output_dir = os.path.join(working_dir, "subnetworks", "kmeans", f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

# Load concatenated data
full_concat = os.path.join(network_dir, "resting_state", 
                          f"sub-{subject}_all-tasks_concatenated_cleaned_smoothed_2.55_fsLR.dtseries.nii")
print(f"Loading: {full_concat}")
all_data_concat = RR.load_data(full_concat)
dtseries_cortex = all_data_concat[:, :64984]

# Load FPN mask
FPN_file = os.path.join(network_dir, 'resting_state', 'Frontoparietal_roi.dscalar.nii')
fpn_mask = np.squeeze(RR.load_data(FPN_file))[:64984].astype(bool) # Only cortex
n_fpn = int(fpn_mask.sum())
print(f"FPN vertices: {n_fpn}")

fpn_indices = np.where(fpn_mask)[0]
FPN_data = dtseries_cortex[:, fpn_mask]

# Z-score and L2-normalize
Z_tv = RR.z_score_np(FPN_data)
X = Z_tv.T
row_norms = np.linalg.norm(X, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
X = X / row_norms
print("Applied L2 normalization for spherical k-means")

# Run k-means clustering
n_clusters = list(range(2, 11))
output_dscalar = os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices.dscalar.nii")

cluster_results, inertia, silhouette_scores, kmeans_list = RR.kmeans_standard(
    n_clusters, 
    X, 
    save_to_file=False,  # Don't save automatically
    remap_to_verts=True,
    filename=output_dscalar, 
    mask_file=FPN_file, 
    dtseries_template=full_concat
)

# Convert to 1-based labels (sklearn returns 0-based)
cluster_results_one_based = cluster_results + 1

# Save all k as dtseries (each k is a "timepoint")
output_dtseries = os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices.dtseries.nii")

# Load full FPN mask (91k space) to get correct indices
fpn_mask_full = np.squeeze(RR.load_data(FPN_file)).astype(bool)
fpn_indices_full = np.where(fpn_mask_full)[0]

# Create full brain array (n_clusters × 91282 grayordinates - cortex + subcortex)
all_k_full = np.zeros((len(n_clusters), 91282), dtype=np.float32)
for i, k in enumerate(n_clusters):
    print(f"{cluster_results_one_based}")
    # Use 1-based labels and full 91k indices
    all_k_full[i, fpn_indices_full] = cluster_results_one_based[i, :]

# Save as dtseries using the template
RR.nib_save(output_dtseries, all_k_full, full_concat)
print(f"Saved dtseries with all k solutions: {output_dtseries}")

# Plots
RR.elbow_plot(n_clusters, inertia, os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices_elbow.png"))
RR.silhouette_plot(silhouette_scores, os.path.join(output_dir, f"sub-{subject}_kmeans_on_vertices_silhouette.png"))

# Metrics CSV
csv_path = os.path.join(output_dir, f"sub-{subject}_clustering_of_kmeans_metrics.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric'] + [f'k={k}' for k in n_clusters])
    writer.writerow(['Inertia'] + inertia)
    writer.writerow(['Silhouette'] + silhouette_scores)
print(f"Saved: {csv_path}")

print("Done!")

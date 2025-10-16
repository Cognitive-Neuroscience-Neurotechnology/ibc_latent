"""
(1)
Clusters on the basis of coherent communities from infomap & their connectivity to LSNs using kmeans
(analogous to cluster_coherent_communities_kmeans.py)
-> similar to infomap approach, but now using kmeans, 
    i.e. clustering FPN communities (from Infomap) based on their connectivity profiles (correlations to other networks)
"""

import os
import argparse
import sys
import numpy as np
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
from scipy.stats import spearmanr
import nibabel as nib
import subprocess

parser = argparse.ArgumentParser(description='Cluster FPN communities using k-means')

parser.add_argument('--subject', help='do not include sub- prefix', required=True)
parser.add_argument('--dir', help='Base working directory', required=True)
args = parser.parse_args()

subject = args.subject.zfill(2)
working_dir = args.dir

"""
00. Set up directories & load data (1. dtseries and 2. atlas with network labels)
"""
print("----- 00. Set up directories & load data -----")

subdir = os.path.join(working_dir, 'individual_networks', f'sub-{subject}')
half_dir = os.path.join(subdir, 'resting_state')
subnetworks_dir = os.path.join(working_dir, 'subnetworks', 'infomap', f'sub-{subject}')
os.makedirs(subnetworks_dir, exist_ok=True)

print("Loading dtseries and atlas as arrays...")

# 1. Load dtseries (concatenated resting-state time series)

#???? Instead take the smoothed one?
#???? Is this the correct way to restrict to cortex only?

#filename = os.path.join(half_dir, f'sub-{subject}_all-tasks_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii
filename = os.path.join(half_dir, f'sub-{subject}_all-tasks_concatenated_cleaned_fsLR.dtseries.nii')
dtseries_img = nib.load(filename)
dtseries_full = dtseries_img.get_fdata()  # shape: (timepoints, 91282)
dtseries_data = dtseries_full[:, :64984]  # cortex-only
print(f"Original dtseries shape: {dtseries_img.shape}")
print(f"Restricted dtseries shape (cortex only): {dtseries_data.shape}")

# 2. Load atlas with network labels (communities)
networks_file = os.path.join(half_dir, 'Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii')
atlas_img = nib.load(networks_file)
atlas_data = atlas_img.get_fdata() # shape: (1, 91282)
print(f"Atlas shape before restriction: {atlas_img.shape}")
atlas_data = atlas_data[:, :64984].flatten()  # Ensure atlas_data is 1D
print(f"Atlas shape after restricting to cortex: {atlas_data.shape}")
print(f"Number of total network labels: {len(np.unique(atlas_data))}")
print(f"Number of unique networks: {len(np.unique(atlas_data)) - 1}")  # -1 for background


"""
01. Parcellate RSNs: create matrix (t timepoints x 19? networks)
"""
print("----- 01. Parcellating dtseries using label data -----")

# Define paths where to save parcellated data (as before in individual_networks/resting_state/)
template_path = os.path.join(half_dir, f'sub-{subject}_ptseries_template.nii')
parcellated_path = os.path.join(half_dir, f'sub-{subject}_individual_nets_concat.ptseries.nii')

# Use wb_command to write a valid ptseries (t x ~19 labels)
print("Parcellating with Workbench...")
cmd = [
    "wb_command",
    "-cifti-parcellate",
    filename,           # dtseries input (full 91k is fine)
    networks_file,      # dlabel with ~21 networks
    "COLUMN",
    parcellated_path
]
subprocess.run(cmd, check=True)

# Load the parcellated result
ptseries_img = nib.load(parcellated_path)
ptseries_data = ptseries_img.get_fdata()         # shape: (time, n_labels)
print("ptseries_data shape:", ptseries_data.shape)

# Use Workbench output as the canonical parcellated data for the rest of the script
all_data_concat = ptseries_data

# Remove FPN and Noise (update indices from label table if possible)
frontoparietal_tseries = np.delete(ptseries_data, [8, -1], axis=1)
print("Removed FPN and Noise. New shape:", frontoparietal_tseries.shape)

# ------------------------------------

"""
02. Get timeseries from infomap communities & correlate with LSNs
"""
print("----- 02. Extract timeseries per FPN community & correlate with other LSNs -----")

subnetworks_file = os.path.join(subnetworks_dir, 'FPN_communities.dscalar.nii')

# Load subnetwork labels (dscalar)
subnetworks = RR.load_data(subnetworks_file)
subnetworks = np.squeeze(subnetworks)  # ensure 1D if possible
print(f"Subnetworks array shape: {subnetworks.shape}")
unique_vals = np.unique(subnetworks[subnetworks != 0])
print(f"Number of communities: {len(unique_vals)})")
communities = unique_vals
n_communities = len(communities)
if n_communities < 2:
    raise RuntimeError(f"Only {n_communities} community label(s) found in {subnetworks_file}. "
                       f"Check you’re using the multi-community dscalar and correct map index.")

corr_matrix = np.zeros((all_data_concat.shape[1], n_communities))
print(f'Shape of correlation matrix (networks x FPN commmunities): {corr_matrix.shape}')

# Loop through communities (skip unassigned!)
ignore_values=[]
ids=[]

for i, SN in enumerate(communities):
    # Build mask for this community and align shapes
    mask = RR.create_mask(subnetworks, SN)   # often returns shape (1, 91282) or (91282,)
    mask = np.squeeze(mask)
    if mask.ndim != 1:
        raise ValueError(f"Unexpected mask ndim={mask.ndim} for label {SN}")

    # Choose dtseries to match mask length
    if mask.shape[0] == dtseries_full.shape[1]:
        data_for_net = dtseries_full
    elif mask.shape[0] == dtseries_data.shape[1]:
        data_for_net = dtseries_data
    elif mask.shape[0] == 91282 and dtseries_data.shape[1] == 64984:
        # restrict mask to cortex to match dtseries_data
        mask = mask[:64984]
        data_for_net = dtseries_data
    else:
        raise ValueError(f"Mask length {mask.shape[0]} does not match dtseries "
                         f"(64984 or 91282).")

    vox_in_net = int((mask != 0).sum())
    print(f'Found {vox_in_net} vertices in subnetwork {SN}...')
    if vox_in_net == 0:
        print(f'Subnetwork {SN} is empty. Skipping...')
        ignore_values.append(SN)
        continue

    # Average timeseries of this community
    if isinstance(data_for_net, np.memmap):
        data_for_net = np.asarray(data_for_net)
    mask_bool = mask.astype(bool)
    subnetwork_tseries = RR.get_network(data_for_net, mask_bool, remove_rest=True)
    average_tseries = np.nanmean(subnetwork_tseries, axis=1)
    ids.append(SN)

    # Correlate with LSN ptseries (use ptseries_data you computed via Workbench)
    corr_matrix_temp, _ = spearmanr(all_data_concat.T, average_tseries, axis=1)
    correlations_with_column_vector = corr_matrix_temp[:-1, -1]
    corr_matrix[:, i] = correlations_with_column_vector

# remove columns with all zeros
corr_matrix = corr_matrix[:, ~(corr_matrix == 0).all(axis=0)]
print('Removed non-existing subnetworks (all zero columns). Resulting shape:', corr_matrix.shape)

# additionally, not all subjects have all LSNs, so we need to remove those as well
corr_matrix = corr_matrix[~((corr_matrix == 0).all(axis=1) | np.isnan(corr_matrix).all(axis=1))]
print('Removed non-existing LSNs (all zero rows). Resulting shape:', corr_matrix.shape)


"""
03. K-means clustering of communities based on their connectivity profiles to LSNs
"""
print("----- 03. K-means clustering of communities -----")

# matrix is 19 LSNs x ~10 SNs ?????????
# This means corr_matrix rows correspond to LSNs (large-scale networks) and columns to SNs (subnetworks/communities).
# The shape comes from removing FPN and Noise columns (so 19 LSNs remain), and the number of SNs depends on the subject's detected communities in FPN (~10).
corr_matrix_SNs = corr_matrix.T  
distance_matrix = 1 - corr_matrix_SNs

# standardise for better clustering (for later)
scaler = StandardScaler()
distance_matrix = scaler.fit_transform(distance_matrix)

output_file=os.path.join(subnetworks_dir, f'{subject}_FPN_infomap_communities_kmeans.dtseries.nii')
n_clusters=range(1,corr_matrix_SNs.shape[0]+1) # made robust, bc different numbers of SNs present

cluster_results, inertia, silhouette_scores, kmeans_list = RR.kmeans_standard(
    n_clusters, 
    distance_matrix, 
    save_to_file=True, remap_to_verts=True, filename=output_file, mask_file=subnetworks_file, dtseries_template=filename,
    ignore_values=ignore_values,
    ids=ids
)

elbow_file=os.path.join(subnetworks_dir, f'{subject}_FPN_infomap_communities_kmeans_elbow_plot.png')
RR.elbow_plot(n_clusters, inertia, elbow_file)

silhouette_file=os.path.join(subnetworks_dir, f'{subject}_FPN_infomap_communities_kmeans_silhouette_plot.png')
RR.silhouette_plot(silhouette_scores, silhouette_file)

for k in n_clusters:
    print('')
    print("Number of clusters:", k)
    labels = cluster_results[k-1, :]

    entropy = RR.compute_entropy(labels)
    bic = RR.compute_bic(kmeans_list[k-1], distance_matrix)
    smallest_size = RR.smallest_cluster_size(labels)

    print("Entropy:", entropy)
    print("BIC:", bic)
    print("Smallest Cluster Size:", smallest_size)


label_table = os.path.join(working_dir, 'subnetworks', 'label_table_infomap_kmeans.txt')
out_dlabel=os.path.join(subnetworks_dir, f'{subject}_FPN_infomap_communities_kmeans.dlabel.nii')
RR.write_dlabel(input_cifti=output_file, label_table=label_table, out_dlabel=out_dlabel)
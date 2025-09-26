"""
Clusters on the basis of coherent communities from infomap & their connectivity to LSNs using kmeans
analogous to cluster_coherent_communities_kmeans.py
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
parser.add_argument('--dir', help='Base working directory, e.g. /ptmp/hmueller2/Downloads', required=True)
args = parser.parse_args()

subject = args.subject.zfill(2)
working_dir = args.dir

"""
00. set up directories
"""

# Adapted paths to match pfm_test_resting.m and get_FPN_communities.py conventions
subdir = os.path.join(working_dir, 'individual_networks', f'sub-{subject}')
half_dir = os.path.join(subdir, 'resting_state')
subnetworks_dir = os.path.join(working_dir, 'subnetworks', 'infomap', f'sub-{subject}')
os.makedirs(subnetworks_dir, exist_ok=True)

# dtseries for subnetworks (concatenated resting-state time series)
filename = os.path.join(half_dir, f'sub-{subject}_all-tasks_concatenated_cleaned_fsLR.dtseries.nii')

# parcellate RSNs
# parc_filename = os.path.join(half_dir, f'sub-{subject}_individual_nets_concat.LR.32k.ptseries.nii') #this will be created
networks_file = os.path.join(half_dir, 'Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii')
print("Loading dtseries and atlas as arrays...")

# Load dtseries and atlas as nibabel images
dtseries_img = nib.load(filename)
atlas_img = nib.load(networks_file)

# Get data arrays
dtseries_data = dtseries_img.get_fdata()  # shape: (timepoints, 91282)
atlas_data = atlas_img.get_fdata().squeeze()[:64984]  # shape: (64984,)
print("atlas_data shape after restricting to cortex:", atlas_data.shape)

# Restrict dtseries to cortex vertices (first 64984)
dtseries_cortex = dtseries_data[:, :64984]
print("dtseries_cortex shape:", dtseries_cortex.shape)
# Now parcellate
print("Parcellating using numpy arrays...")
all_data_concat = RR.cifti_parcellate(dtseries_cortex, atlas_data)
print("all_data_concat shape:", all_data_concat.shape) # 30 i guess

all_data_concat = np.delete(all_data_concat, [8, -1], axis=1) # delete FPN (8) and Noise (last)
print("Removed the FPN and Noise. New shape:", all_data_concat.shape) # should be 28 columns i guess

# Save parcellated data as before
save_path = os.path.join(half_dir, f'sub-{subject}_individual_nets_concat.LR.32k.ptseries.nii')
dscalar_template, _ = RR.wb_label_to_roi(networks_file, half_dir, 'Frontoparietal')

# Paths for input dtseries, atlas, and output ptseries
dtseries_path = filename
atlas_path = networks_file
ptseries_path = os.path.join(half_dir, f'sub-{subject}_individual_nets_concat.ptseries.nii')

# Run Workbench parcellation if ptseries file does not exist
if not os.path.exists(ptseries_path):
    print(f"Running wb_command -cifti-parcellate for subject {subject} ...")
    cmd = [
        "wb_command",
        "-cifti-parcellate",
        dtseries_path,
        atlas_path,
        "COLUMN",
        ptseries_path
    ]
    subprocess.run(cmd, check=True)
else:
    print(f"ptseries file already exists for subject {subject}: {ptseries_path}")

# Now save parcellated data using the ptseries template
RR.nib_save(ptseries_path, all_data_concat, ptseries_path)

# concatenate original dtseries file for the subnetwork tseries
dtseries_concat=RR.load_data(filename)

subnetworks_file = os.path.join(subnetworks_dir, 'FPN_communities.dscalar.nii')

# get infomap communities
subnetworks = RR.load_data(subnetworks_file)
communities = np.unique(subnetworks[subnetworks != 0])
n_communities = len(communities)

corr_matrix = np.zeros((all_data_concat.shape[1], n_communities))
print(corr_matrix.shape)

# loop through communities (skip unassigned!)
ignore_values=[]
ids=[]

for i, SN in enumerate(communities):
    # get the subnetwork data
    mask=RR.create_mask(subnetworks, SN)
    print(f'Found {mask.sum()} vertices in subnetwork {SN}...')

    if mask.sum() == 0:
        print(f'Subnetwork {SN} is empty. Skipping...')
        ignore_values.append(SN)
        continue

    # get average timeseries for this subnetwork
    subnetwork_tseries=RR.get_network(dtseries_concat, mask, remove_rest=True)
    average_tseries=np.mean(subnetwork_tseries, axis=1)
    ids.append(SN)

    print("Computing correlation matrix...")
    corr_matrix_temp, _ = spearmanr(all_data_concat.T, average_tseries, axis=1)
    correlations_with_column_vector = corr_matrix_temp[:-1, -1]
    print(correlations_with_column_vector.shape)
    corr_matrix[:, i] = correlations_with_column_vector

# remove columns with all zeros
corr_matrix = corr_matrix[:, ~(corr_matrix == 0).all(axis=0)]
print('Removed non-existing subnetworks (all zero columns). Resulting shape:', corr_matrix.shape)

# additionally, not all subjects have all LSNs, so we need to remove those as well
corr_matrix = corr_matrix[~((corr_matrix == 0).all(axis=1) | np.isnan(corr_matrix).all(axis=1))]
print('Removed non-existing LSNs (all zero rows). Resulting shape:', corr_matrix.shape)

"""
K-means
"""

# matrix is 19 LSNs x ~10 SNs
# This means corr_matrix rows correspond to LSNs (large-scale networks) and columns to SNs (subnetworks/communities).
# The shape comes from removing FPN and Noise columns (so 19 LSNs remain), and the number of SNs depends on the subject's detected communities (~10).
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
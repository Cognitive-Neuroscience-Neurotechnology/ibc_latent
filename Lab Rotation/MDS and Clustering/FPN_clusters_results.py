import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image, surface, datasets
import hcp_utils as hcp
import nibabel as nib
import os
from sklearn.metrics.pairwise import cosine_similarity

def plot_brain_surface(cortex_data, subject, output_path, min_thresh=1, max_thresh=2, cm='gist_rainbow'):
    fig = plt.figure(figsize=[20, 10])
    ax = fig.add_subplot(1, 6, 1, projection='3d')
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated, 
        cortex_data, 
        hemi='left', 
        view='lateral', 
        colorbar=False,
        threshold=min_thresh, 
        vmax=max_thresh, 
        bg_map=hcp.mesh.sulc, 
        bg_on_data=True, 
        darkness=0.3, 
        axes=ax, 
        figure=fig, 
        cmap=cm, 
        symmetric_cbar=True
    )
    ax = fig.add_subplot(1, 6, 2, projection='3d')
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated, 
        cortex_data, 
        hemi='right', 
        view='lateral', 
        colorbar=False,
        threshold=min_thresh, 
        vmax=max_thresh, 
        bg_map=hcp.mesh.sulc, 
        bg_on_data=True, 
        darkness=0.3, 
        axes=ax, 
        figure=fig, 
        cmap=cm, 
        symmetric_cbar=True
    )
    ax = fig.add_subplot(1, 6, 3, projection='3d')
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated, 
        cortex_data, 
        view='dorsal', 
        colorbar=False,
        threshold=min_thresh, 
        vmax=max_thresh, 
        bg_map=hcp.mesh.sulc, 
        bg_on_data=True, 
        darkness=0.3, 
        axes=ax, 
        figure=fig, 
        cmap=cm, 
        symmetric_cbar=True
    )
    ax = fig.add_subplot(1, 6, 4, projection='3d')
    plotting.plot_surf_stat_map(
        hcp.mesh.inflated, 
        cortex_data, 
        view='anterior', 
        colorbar=False,
        threshold=min_thresh, 
        vmax=max_thresh, 
        bg_map=hcp.mesh.sulc, 
        bg_on_data=True, 
        darkness=0.3, 
        axes=ax, 
        figure=fig, 
        cmap=cm, 
        symmetric_cbar=True
    )
    ax = fig.add_subplot(1, 6, 5, projection='3d')
    plotting.plot_surf_stat_map(
        hcp.mesh.very_inflated_left, 
        hcp.right_cortex_data(hcp.unparcellate(Xp, hcp.mmp)), 
        hemi='left', 
        view='medial', 
        colorbar=False,
        threshold=min_thresh, 
        vmax=max_thresh, 
        bg_map=hcp.mesh.sulc_left, 
        bg_on_data=True, 
        darkness=0.3, 
        axes=ax, 
        figure=fig, 
        cmap=cm, 
        symmetric_cbar=True
    )
    ax = fig.add_subplot(1, 6, 6, projection='3d')
    plotting.plot_surf_stat_map(
        hcp.mesh.very_inflated_right, 
        hcp.right_cortex_data(hcp.unparcellate(Xp, hcp.mmp)),
        hemi='right', 
        view='medial', 
        colorbar=False,
        threshold=min_thresh, 
        vmax=max_thresh, 
        bg_map=hcp.mesh.sulc_right, 
        bg_on_data=True, 
        darkness=0.3, 
        axes=ax, 
        figure=fig, 
        cmap=cm, 
        symmetric_cbar=True
    )
    # Save the figure
    fig.savefig(output_path)
    plt.close(fig)

def average_rsms_within_clusters(subject, clusters, base_rsm_path):
    cluster_rsms = {}
    for cluster in clusters:
        rsm_files = [os.path.join(base_rsm_path, f'rsm_{parcel}_{subject}_all-contrasts_method-cosine.csv') for parcel in clusters[cluster]]
        rsms = [pd.read_csv(rsm_file, index_col=0).values for rsm_file in rsm_files]
        cluster_rsms[cluster] = np.mean(rsms, axis=0)
    return cluster_rsms

def compute_elementwise_euclidean_distance(cluster_rsm1, cluster_rsm2):
    distance_matrix = np.zeros(cluster_rsm1.shape)
    for i in range(cluster_rsm1.shape[0]):
        for j in range(cluster_rsm1.shape[1]):
            distance_matrix[i, j] = np.linalg.norm(cluster_rsm1[i, j] - cluster_rsm2[i, j])
    return distance_matrix

def save_pairwise_similarity(cluster_rsms, subject, output_dir):
    cluster_ids = list(cluster_rsms.keys())
    for i, cluster1 in enumerate(cluster_ids):
        for j, cluster2 in enumerate(cluster_ids):
            if i < j:
                distance_matrix = compute_elementwise_euclidean_distance(cluster_rsms[cluster1], cluster_rsms[cluster2])
                distance_df = pd.DataFrame(distance_matrix)
                output_path = os.path.join(output_dir, f'{cluster1}-{cluster2}_{subject}.csv')
                distance_df.to_csv(output_path, index=False)

### MAIN ###

run = 'run_10'
base_path = '/home/hmueller2/ibc_code/ibc_output_KMeans_onMDS/'
rsm_base_path = '/home/hmueller2/ibc_code/ibc_output_RSA_cosine/'

# Get all subjects from the filenames in the folder
subjects = [f.split('_')[1].split('.')[0] for f in os.listdir(os.path.join(base_path, run)) if f.startswith('parcel-cluster_') and f.endswith('.csv')]

for subject in subjects:
    print(f'Processing {subject}')
    # Load the parcel_clusters_df DataFrame
    parcel_clusters_path = os.path.join(base_path, run, f'parcel-cluster_{subject}.csv')
    if not os.path.exists(parcel_clusters_path):
        print(f"File not found: {parcel_clusters_path}")
        continue

    parcel_clusters_df = pd.read_csv(parcel_clusters_path)

    # Initialize Xp with zeros
    Xp = np.zeros(len(hcp.mmp.labels))

    # Fill Xp using the "index_Xp" and "cluster" columns from parcel_clusters_df
    for _, row in parcel_clusters_df.iterrows():
        Xp[int(row['index_Xp'])] = row['cluster']

    # Remove the first number from Xp
    Xp = Xp[1:380]

    # Map values to surface vertices
    surface_data = hcp.unparcellate(Xp, hcp.mmp)
    cortex_data = hcp.cortex_data(surface_data)

    # Define output path for brain surface plot
    output_path = os.path.join(base_path, run, f'cluster-brain_{subject}.png')

    # Plot and save the brain surface
    print(f'Saving brain surface plot to {output_path}')
    plot_brain_surface(cortex_data, subject, output_path)

    # Group parcels by cluster
    clusters = parcel_clusters_df.groupby('cluster')['parcel_label'].apply(list).to_dict()

    # Average RSMs within clusters
    cluster_rsms = average_rsms_within_clusters(subject, clusters, os.path.join(rsm_base_path, f'{subject}'))

    # Save pairwise similarity between clusters
    print(f'Saving pairwise similarity for {subject}')
    save_pairwise_similarity(cluster_rsms, subject, os.path.join(base_path, run))

print('Processing complete.')
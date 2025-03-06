import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image, surface, datasets
import hcp_utils as hcp
import nibabel as nib
import os

run = 'run_06'
base_path = '/home/hmueller2/ibc_code/ibc_output_KMeans_onMDS/'

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

    min_thresh = 0.1 # Keep this above 0 so that all non-FPN parcels are transparent
    max_thresh = 2 # Value should be number of clusters
    cm = 'plasma'

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
        hcp.left_cortex_data(cortex_data), 
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
        hcp.right_cortex_data(cortex_data), 
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
    output_path = os.path.join(base_path, run, f'cluster-brain_{subject}.png')
    fig.savefig(output_path)
    plt.close(fig)
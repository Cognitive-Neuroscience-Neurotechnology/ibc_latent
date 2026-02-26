import os
import numpy as np
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import time
from nibabel.freesurfer.io import read_annot
from task_contrasts import task_contrasts
from config_ICA import base_dir, output_dir
import seaborn as sns
import matplotlib.pyplot as plt

def load_contrast_map(file_path):
    try:
        img = nib.load(file_path)
        data = np.array([darray.data for darray in img.darrays])
        data = data.squeeze()  # Remove any singleton dimensions
        return data
    except FileNotFoundError:
        return None

def extract_parcel_data(data, parcel_mapping):
    parcel_data = {}
    for vertex, parcel in parcel_mapping.items():
        parcel_name = parcel.decode('utf-8')
        if parcel_name not in parcel_data:
            parcel_data[parcel_name] = []
        try:
            parcel_data[parcel_name].append(data[vertex])
        except IndexError as e:
            print(f"IndexError: {e} for vertex {vertex} and parcel {parcel_name} with data shape {data.shape}")
            continue
    
    # Convert lists to numpy arrays
    for parcel_name in parcel_data:
        parcel_data[parcel_name] = np.array(parcel_data[parcel_name])
    return parcel_data

def average_sessions(file_paths):
    data_sessions = [load_contrast_map(fp) for fp in file_paths]
    data_sessions = [data for data in data_sessions if data is not None]
    if not data_sessions:
        return None
    average_data = np.mean(data_sessions, axis=0)
    return average_data

def create_contrast_parcel_matrix(subject, base_dir, task_contrasts, lh_parcel_mapping, rh_parcel_mapping):
    """Create a matrix (contrast x parcel) for each subject."""
    session_dirs = [d for d in os.listdir(os.path.join(base_dir, f'sub-{subject}')) if d.startswith('ses-')]
    
    contrast_maps = []
    for hemisphere in ['lh', 'rh']:
        for task, contrasts in task_contrasts.items():
            for contrast in contrasts:
                file_paths = [
                    os.path.join(
                        base_dir, 
                        f'sub-{subject}', 
                        session, 
                        f'sub-{subject}_ses-{session.split("-")[1]}_task-{task}_dir-ffx_space-fsaverage7_hemi-{hemisphere}_ZMap-{contrast}.gii'
                    ) 
                    for session in session_dirs
                ]
                
                # Check if files exist
                file_paths = [fp for fp in file_paths if os.path.exists(fp)]
                
                if not file_paths:
                    continue
                
                # Average the data across sessions if there are multiple sessions
                if len(file_paths) > 1:
                    data = average_sessions(file_paths)
                else:
                    data = load_contrast_map(file_paths[0])
                
                if data is not None:
                    contrast_maps.append((contrast, data, hemisphere))
    
    if not contrast_maps:
        print(f"No contrast maps found for subject {subject}. Skipping...")
        return None
    
    # Extract parcel data
    parcel_data = {}
    for contrast, data, hemisphere in contrast_maps:
        if contrast not in parcel_data:
            parcel_data[contrast] = {}
            
        if hemisphere == 'lh':
            lh_data = extract_parcel_data(data, lh_parcel_mapping)
            parcel_data[contrast].update(lh_data)
        elif hemisphere == 'rh':
            rh_data = extract_parcel_data(data, rh_parcel_mapping)
            parcel_data[contrast].update(rh_data)
    
    # Create the contrast x parcel matrix
    parcels = list(parcel_data[contrast].keys())
    contrasts = list(parcel_data.keys())
    matrix = np.zeros((len(contrasts), len(parcels)))
    for i, contrast in enumerate(contrasts):
        for j, parcel in enumerate(parcels):
            matrix[i, j] = np.mean(parcel_data[contrast][parcel])
    
    print(f"Matrix for subject {subject} created with shape {matrix.shape}")
    return matrix, parcels

def perform_pca(matrix, n_components):
    """Perform Principal Component Analysis (PCA) on the provided matrix."""
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(matrix)
    return reduced_matrix, pca

def perform_ica(matrix, n_components):
    """Perform Independent Component Analysis (ICA) on the provided matrix."""
    ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
    components = ica.fit_transform(matrix)  # Shape (n_samples, n_components)
    return components, ica

def visualize_components_heatmap(components, matrix, title, output_dir, parcel_names):
    """Visualize the ICA components mapped to parcels as a heatmap."""
    # Get the mapping from components to parcels
    # For FastICA, we need to use the components_ attribute from the model
    ica = FastICA(n_components=components.shape[1])
    ica.fit(matrix)
    component_parcel_mapping = ica.components_  # Shape (n_components, n_parcels)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(component_parcel_mapping,
                xticklabels=parcel_names, 
                yticklabels=[f'Component {i+1}' for i in range(components.shape[1])],
                cmap='viridis', 
                annot=False)  # Set annot=False for better visibility with many parcels
    plt.title(f'{title} - ICA Components by Parcels')
    plt.xlabel('Parcels')
    plt.ylabel('Components')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_components_parcels_heatmap.png'))
    plt.close()

def visualize_components_heatmap(components, title, output_dir, parcel_names):
    """Visualize the ICA components as a heatmap and save the plot."""
    plt.figure(figsize=(15, 10))
    sns.heatmap(components, xticklabels=parcel_names, yticklabels=[f'Component {i+1}' for i in range(components.shape[1])], cmap='viridis', annot=True)
    plt.title(f'{title} - ICA Components Heatmap')
    plt.xlabel('Parcels')
    plt.ylabel('Components')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_components_heatmap.png'))
    plt.close()

def visualize_component_clusters(components, title, output_dir, parcel_names):
    """Visualize hierarchical clustering of parcels based on ICA components."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # Perform hierarchical clustering
    linked = linkage(components.T, 'ward')  # Transpose to get parcels as observations
    
    plt.figure(figsize=(18, 10))
    dendrogram(linked, 
               orientation='top',
               labels=parcel_names,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title(f'{title} - Hierarchical Clustering of Parcels by Component Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_parcels_dendrogram.png'))
    plt.close()

def main(n_pca_components, n_ica_components):
    # Load network partition and filter for frontoparietal network parcels
    network_partition_path = '/home/hmueller2/Downloads/FPN_parcellation_cole/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
    network_partition = pd.read_csv(network_partition_path, sep='\t')
    fpn_parcels = network_partition[network_partition['NETWORKKEY'] == 7]
    fpn_parcels_names = fpn_parcels['GLASSERLABELNAME'].dropna().tolist()
    
    # Load annotation files
    lh_annot_file = '/home/hmueller2/Downloads/atlas_glasser/glasser_fsaverage/3498446/lh.HCP-MMP1.annot'
    rh_annot_file = '/home/hmueller2/Downloads/atlas_glasser/glasser_fsaverage/3498446/rh.HCP-MMP1.annot'
    labels_lh, ctab_lh, names_lh = read_annot(lh_annot_file)
    labels_rh, ctab_rh, names_rh = read_annot(rh_annot_file)
    
    # Create vertex-to-parcel mappings for frontoparietal network parcels
    vertices_lh = np.arange(len(labels_lh))
    vertices_rh = np.arange(len(labels_rh))
    lh_parcel_mapping = {vertex: names_lh[label] for vertex, label in zip(vertices_lh, labels_lh)}
    rh_parcel_mapping = {vertex: names_rh[label] for vertex, label in zip(vertices_rh, labels_rh)}
    fpn_parcels_lh_mapping = {vertex: lh_parcel_mapping[vertex] for vertex in vertices_lh if lh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
    fpn_parcels_rh_mapping = {vertex: rh_parcel_mapping[vertex] for vertex in vertices_rh if rh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
    
    # Process each subject
    subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')]
    for subject in subjects:
        print(f"Processing subject {subject}...")
        matrix, parcels = create_contrast_parcel_matrix(subject, base_dir, task_contrasts, fpn_parcels_lh_mapping, fpn_parcels_rh_mapping)
        if matrix is not None:
            # Apply PCA to reduce the number of contrasts
            reduced_matrix, pca = perform_pca(matrix, n_pca_components)
            print(f"PCA reduced matrix for subject {subject} with shape {reduced_matrix.shape} (reduced contrasts × parcels)")
            
            # Apply ICA to the reduced matrix
            components, ica = perform_ica(reduced_matrix.T, n_ica_components)
            components_T = components.T  # Shape (n_components, n_parcels)
            
            # Create subfolder in output_dir based on the number of components
            components_output_dir = os.path.join(output_dir, f'{n_ica_components}_components')
            os.makedirs(components_output_dir, exist_ok=True)
            
            # Save ICA components
            np.save(os.path.join(components_output_dir, f'sub-{subject}_ica_components.npy'), components_T)
            print(f"ICA components for subject {subject} with shape {components_T.shape} (components × parcels) saved to {components_output_dir}")
            
            # Visualize and save the components as heatmap
            visualize_components_heatmap(components_T, f'sub-{subject}', components_output_dir, parcels)
            print(f"ICA component heatmap for subject {subject} saved to {components_output_dir}")
            
            # Add hierarchical clustering visualization
            visualize_component_clusters(components_T, f'sub-{subject}', components_output_dir, parcels)
            print(f"ICA component clustering for subject {subject} saved to {components_output_dir}")

if __name__ == "__main__":
    main(n_pca_components=7, n_ica_components=6)  # Specify the number of PCA and ICA components here
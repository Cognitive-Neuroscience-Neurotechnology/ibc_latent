import os
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import time
from nibabel.freesurfer.io import read_annot
from task_contrasts import task_contrasts
from config_ICA import base_dir, output_dir

def load_contrast_map(file_path):
    try:
        img = nib.load(file_path)
        data = np.array([darray.data for darray in img.darrays])
        data = data.squeeze()  # Remove any singleton dimensions
        #print(f"Loaded contrast map from {file_path} with shape {data.shape}")
        return data
    except FileNotFoundError:
        #print(f"File not found: {file_path}")
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
        if hemisphere == 'lh':
            parcel_data[contrast] = extract_parcel_data(data, lh_parcel_mapping)
        elif hemisphere == 'rh':
            parcel_data[contrast] = extract_parcel_data(data, rh_parcel_mapping)
    
    # Create the contrast x parcel matrix
    parcels = list(parcel_data[contrast].keys())
    contrasts = list(parcel_data.keys())
    matrix = np.zeros((len(contrasts), len(parcels)))
    for i, contrast in enumerate(contrasts):
        for j, parcel in enumerate(parcels):
            matrix[i, j] = np.mean(parcel_data[contrast][parcel])
    
    print(f"Matrix for subject {subject} created with shape {matrix.shape}")
    return matrix

def perform_ica(matrix, n_components):
    """Perform Independent Component Analysis (ICA) on the provided matrix."""
    ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
    components = ica.fit_transform(matrix)
    return components

def visualize_components(components, title, output_dir):
    """Visualize the ICA components and save the plots."""
    n_components = components.shape[1]
    plt.figure(figsize=(15, 5))
    for i in range(n_components):
        plt.subplot(1, n_components, i + 1)
        plt.plot(components[:, i])
        plt.title(f'{title} - Component {i + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_components.png'))
    plt.close()

def main():
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
    
    # Create vertex-to-parcel mappings for frontoparietal parcels
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
        matrix = create_contrast_parcel_matrix(subject, base_dir, task_contrasts, fpn_parcels_lh_mapping, fpn_parcels_rh_mapping)
        if matrix is not None:
            n_components = min(matrix.shape[0], 3)  # Set the number of components
            
            # Create subfolder in output_dir based on the number of components
            components_output_dir = os.path.join(output_dir, f'{n_components}_components')
            os.makedirs(components_output_dir, exist_ok=True)
            
            components = perform_ica(matrix, n_components)
            
            # Save ICA components
            np.save(os.path.join(components_output_dir, f'sub-{subject}_ica_components.npy'), components)
            print(f"ICA components for subject {subject} saved to {components_output_dir}")
            
            # Visualize and save the components
            visualize_components(components, f'sub-{subject}', components_output_dir)
            print(f"ICA component plots for subject {subject} saved to {components_output_dir}")

if __name__ == "__main__":
    main()
import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from nibabel.freesurfer.io import read_annot
from config_RSA import base_dir, output_dir, method

# Ensure base_dir and output_dir are not empty
if not base_dir or not output_dir:
    raise ValueError("base_dir and output_dir must be defined in config_RSA.py")
from task_contrasts import task_contrasts

# Load network_partition (.txt)
network_partition_path = '/home/hmueller2/Downloads/Cole_FPN_Parcellation/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
network_partition = pd.read_csv(network_partition_path, sep='\t')
print("Network partition loaded.")

# Only keep GLASSERLABELNAMEs of those that are in the frontoparietal network (Networkkey = 7)
fpn_parcels = network_partition[network_partition['NETWORKKEY'] == 7]
fpn_parcels_names = fpn_parcels['GLASSERLABELNAME'].dropna().tolist()
print("Frontoparietal network parcels filtered.")

# Load annot_file
lh_annot_file = '/home/hmueller2/Downloads/Atlas/glasser_fsaverage/3498446/lh.HCP-MMP1.annot'
rh_annot_file = '/home/hmueller2/Downloads/Atlas/glasser_fsaverage/3498446/rh.HCP-MMP1.annot'
labels_lh, ctab_lh, names_lh = read_annot(lh_annot_file)
labels_rh, ctab_rh, names_rh = read_annot(rh_annot_file)
print("Annotation files loaded.")

# Do a vertex-to-parcel mapping for Frontoparietal parcels
vertices_lh = np.arange(len(labels_lh))
vertices_rh = np.arange(len(labels_rh))
lh_parcel_mapping = {vertex: names_lh[label] for vertex, label in zip(vertices_lh, labels_lh)}
rh_parcel_mapping = {vertex: names_rh[label] for vertex, label in zip(vertices_rh, labels_rh)}
fpn_parcels_lh_mapping = {vertex: lh_parcel_mapping[vertex] for vertex in vertices_lh if lh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
fpn_parcels_rh_mapping = {vertex: rh_parcel_mapping[vertex] for vertex in vertices_rh if rh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
print("Vertex-to-parcel mappings created.")


## FUNCTIONS

def load_contrast_map(file_path):
    img = nib.load(file_path)
    data = np.array([darray.data for darray in img.darrays])
    session_number = os.path.basename(file_path).split('_')[1].split('-')[1]
    print(f"Contrast map loaded for session {session_number} and shape {data.shape}.")
    return data

def extract_parcel_data(data, parcel_mapping):
    parcel_data = {}
    for vertex, parcel in parcel_mapping.items():
        parcel_name = parcel.decode('utf-8')
        if parcel_name not in parcel_data:
            parcel_data[parcel_name] = []
        parcel_data[parcel_name].append(data[vertex, :])
    
    # Convert lists to numpy arrays
    for parcel_name in parcel_data:
        parcel_data[parcel_name] = np.array(parcel_data[parcel_name])
    print(f"Parcel data extracted with shape: {parcel_data[parcel_name].shape}")
    return parcel_data

def average_sessions(file_paths):
    data_sessions = [load_contrast_map(fp) for fp in file_paths]
    average_data = np.mean(data_sessions, axis=0)
    print(f"Data averaged across sessions with shape: {average_data.shape}")
    return average_data

def compute_rsm(activations, output_dir, parcel_name, subject, task, contrast):
    # Check if the input is a 2D array
    assert activations.ndim == 2, f"Input activations should be a 2D array, but got {activations.ndim}D array"
    
    n_conditions = activations.shape[1]
    rsm = np.zeros((n_conditions, n_conditions))
    
    for i in range(n_conditions):
        for j in range(n_conditions):
            if method == 'cosine':
                norm_i = np.linalg.norm(activations[:, i])
                norm_j = np.linalg.norm(activations[:, j])
                if norm_i == 0 or norm_j == 0:
                    print(f"Warning: Zero norm encountered for conditions {i} or {j}")
                    rsm[i, j] = 0
                else:
                    rsm[i, j] = np.dot(activations[:, i], activations[:, j]) / (norm_i * norm_j)
            elif method == 'spearman':
                rsm[i, j], _ = spearmanr(activations[:, i], activations[:, j])
            else:
                raise ValueError(f"Unknown method: {method}")
    
    # Check if the output is a 2D array
    assert rsm.ndim == 2, f"Output RSM should be a 2D array, but got {rsm.ndim}D array"
    
    # Print the shape of the RSM
    print(f"RSM shape: {rsm.shape}")
    
    # Save the RSM to a CSV file
    output_file = os.path.join(output_dir, f"rsm_{parcel_name}_sub-{subject}_task-{task}_contrast-{contrast}.csv")
    np.savetxt(output_file, rsm, delimiter=",")
    
    return rsm


## MAIN

print(f"Method: {method}")
subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')]

for subject in subjects:
    print("-" * 100)
    print(f"Processing subject {subject}...")
    subject_output_dir = os.path.join(output_dir, f'sub-{subject}')
    os.makedirs(subject_output_dir, exist_ok=True)
    
    for hemisphere in ['lh', 'rh']:
        # Find all sessions for the current subject
        session_dirs = [d for d in os.listdir(os.path.join(base_dir, f'sub-{subject}')) if d.startswith('ses-')]
        print(f"Session directories for subject {subject}: {session_dirs}")
        
        # Load all contrast maps for all tasks and hemisphere
        contrast_maps = []
        for task, contrasts in task_contrasts.items():
            for contrast in contrasts:
                file_paths = [os.path.join(base_dir, f'sub-{subject}', session, f'sub-{subject}_ses-{session.split("-")[1]}_task-{task}_dir-ffx_space-fsaverage7_hemi-{hemisphere}_ZMap-{contrast}.gii') for session in session_dirs]
                
                # Check if files exist
                file_paths = [fp for fp in file_paths if os.path.exists(fp)]
                if not file_paths:
                    print(f"No files found for subject {subject}, task {task}, contrast {contrast}, hemisphere {hemisphere}. Skipping...")
                    continue
                print(f"Found files for contrast {contrast}: {file_paths}")
                
                # Average the data across sessions if there are multiple sessions
                if len(file_paths) > 1:
                    data = average_sessions(file_paths)
                else:
                    data = load_contrast_map(file_paths[0])
                
                contrast_maps.append(data)
        
        # Check if contrast_maps is not empty
        if not contrast_maps:
            print(f"No contrast maps found for subject {subject}, hemisphere {hemisphere}. Skipping...")
            continue
        
        # Stack contrast maps to form the activations array
        activations = np.stack(contrast_maps, axis=0)
        print(f"Activations shape after stacking: {activations.shape}")

        # Reshape the activations array to (n_voxels, n_conditions)
        activations = activations.squeeze().transpose(1, 0)
        print(f"Activations shape after reshaping: {activations.shape}")
            
        # Extract parcel data
        if hemisphere == 'lh':
            parcel_data = extract_parcel_data(activations, fpn_parcels_lh_mapping)
        elif hemisphere == 'rh':
            parcel_data = extract_parcel_data(activations, fpn_parcels_rh_mapping)
        else:
            raise ValueError(f"Unexpected hemisphere value: {hemisphere}")
            
        # Z-score the activations
        for parcel_name in parcel_data:
            parcel_data[parcel_name] = zscore(parcel_data[parcel_name], axis=0)
            
        # Compute RSM for each parcel using the specified method
        for parcel_name, activations in parcel_data.items():
            # Print the shape of the activations array
            print(f"Activations shape for parcel {parcel_name}: {activations.shape}")
                
            rsm = compute_rsm(activations, subject_output_dir, parcel_name, subject, "all_tasks", "all_contrasts")
            print(f"RSM saved to {subject_output_dir}")
    print("-" * 50 + "Task Done" + "-" * 50) 
    print(f"Processing for subject {subject} completed.")
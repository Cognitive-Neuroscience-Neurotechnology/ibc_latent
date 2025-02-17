import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from nibabel.freesurfer.io import read_annot
import h5py
from config_RSA import base_dir, output_dir
from task_contrasts import task_contrasts

# Load network_partition (.txt)
print("Loading network partition...")
network_partition_path = '/home/hmueller2/Downloads/Cole_FPN_Parcellation/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
network_partition = pd.read_csv(network_partition_path, sep='\t')
print("Network partition loaded.")

# Only keep GLASSERLABELNAMEs of those that are in the frontoparietal network (Networkkey = 7)
print("Filtering frontoparietal network parcels...")
fpn_parcels = network_partition[network_partition['NETWORKKEY'] == 7]
fpn_parcels_names = fpn_parcels['GLASSERLABELNAME'].dropna().tolist()
print("Frontoparietal network parcels filtered.")

# Load annot_file
print("Loading annotation files...")
lh_annot_file = '/home/hmueller2/Downloads/Atlas/glasser_fsaverage/3498446/lh.HCP-MMP1.annot'
rh_annot_file = '/home/hmueller2/Downloads/Atlas/glasser_fsaverage/3498446/rh.HCP-MMP1.annot'

labels_lh, ctab_lh, names_lh = read_annot(lh_annot_file)
labels_rh, ctab_rh, names_rh = read_annot(rh_annot_file)
print("Annotation files loaded.")

# Do a vertex-to-parcel mapping for Frontoparietal parcels
print("Creating vertex-to-parcel mappings...")
vertices_lh = np.arange(len(labels_lh))
vertices_rh = np.arange(len(labels_rh))

lh_parcel_mapping = {vertex: names_lh[label] for vertex, label in zip(vertices_lh, labels_lh)}
rh_parcel_mapping = {vertex: names_rh[label] for vertex, label in zip(vertices_rh, labels_rh)}

fpn_parcels_lh_mapping = {vertex: lh_parcel_mapping[vertex] for vertex in vertices_lh if lh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
fpn_parcels_rh_mapping = {vertex: rh_parcel_mapping[vertex] for vertex in vertices_rh if rh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
print("Vertex-to-parcel mappings created.")

def load_surface_map(file_path):
    print(f"Loading surface map from {file_path}...")
    img = nib.load(file_path)
    data = np.array([darray.data for darray in img.darrays])
    print("Surface map loaded.")
    return data

def extract_parcel_data(data, parcel_mapping):
    print("Extracting parcel data...")
    parcel_data = {}
    for vertex, parcel in parcel_mapping.items():
        parcel_name = parcel.decode('utf-8')
        if parcel_name not in parcel_data:
            parcel_data[parcel_name] = []
        parcel_data[parcel_name].append(data[:, vertex])
    
    # Convert lists to numpy arrays and average activations across vertices
    for parcel_name in parcel_data:
        parcel_data[parcel_name] = np.mean(parcel_data[parcel_name], axis=0)
    print("Parcel data extracted.")
    return parcel_data

def average_sessions(file_paths):
    print("Averaging data across sessions...")
    data_sessions = [load_surface_map(fp) for fp in file_paths]
    average_data = np.mean(data_sessions, axis=0)
    print("Data averaged.")
    return average_data

def compute_rsm_cosine(activations):
    n_conditions = activations.shape[1]
    rsm = np.zeros((n_conditions, n_conditions))
    for i in range(n_conditions):
        for j in range(n_conditions):
            rsm[i, j] = np.dot(activations[:, i], activations[:, j]) / (np.linalg.norm(activations[:, i]) * np.linalg.norm(activations[:, j]))
    return rsm

# Define the parameters to iterate over
print(f"Base directory: {base_dir}")
subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')]
hemispheres = ['lh', 'rh']
nParcels = len(fpn_parcels_names)
n_tasks = len(task_contrasts)

# Initialize RSMs array
rsms_parcels_allsubjs = np.zeros((nParcels, len(subjects), n_tasks, n_tasks))

for subject_idx, subject in enumerate(subjects):
    print(f"Processing subject {subject}...")
    for task_idx, (task, contrasts) in enumerate(task_contrasts.items()):
        for contrast_idx, contrast in enumerate(contrasts):
            for hemisphere in hemispheres:
                print(f"Processing task {task}, contrast {contrast}, hemisphere {hemisphere}...")
                # Find all sessions for the current subject, task, and contrast
                session_dirs = [d for d in os.listdir(os.path.join(base_dir, f'sub-{subject}')) if d.startswith('ses-')]
                file_paths = [os.path.join(base_dir, f'sub-{subject}', session, f'sub-{subject}_ses-{session.split("-")[1]}_task-{task}_dir-ffx_space-fsaverage7_hemi-{hemisphere}_ZMap-{contrast}.gii') for session in session_dirs]
                
                # Check if files exist
                file_paths = [fp for fp in file_paths if os.path.exists(fp)]
                if not file_paths:
                    print(f"No files found for subject {subject}, task {task}, contrast {contrast}, hemisphere {hemisphere}. Skipping...")
                    continue
                print(f"Found files: {file_paths}")
                
                # Average the data across sessions if there are multiple sessions
                if len(file_paths) > 1:
                    data = average_sessions(file_paths)
                else:
                    data = load_surface_map(file_paths[0])
                
                # Extract parcel data
                if hemisphere == 'lh':
                    parcel_data = extract_parcel_data(data, fpn_parcels_lh_mapping)
                elif hemisphere == 'rh':
                    parcel_data = extract_parcel_data(data, fpn_parcels_rh_mapping)
                else:
                    raise ValueError(f"Unexpected hemisphere value: {hemisphere}")
                
                # Compute RSM for each parcel using cosine similarity
                for parcel_idx, (parcel_name, activations) in enumerate(parcel_data.items()):
                    print(f"Computing RSM for parcel {parcel_name}...")
                    rsm = compute_rsm_cosine(activations)
                    rsms_parcels_allsubjs[parcel_idx, subject_idx, task_idx, contrast_idx] = rsm

# Save RSMs to HDF5 file
output_file = os.path.join(output_dir, 'rsms_parcels_allsubjs.h5')
with h5py.File(output_file, 'w') as h5f:
    h5f.create_dataset('data', data=rsms_parcels_allsubjs)
print(f"RSMs saved to {output_file}.")

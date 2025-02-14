import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr

def load_surface_maps(file_paths):
    data = []
    for file_path in file_paths:
        img = nib.load(file_path)
        data.append(img.get_fdata())
    return np.array(data)

def average_data_across_sessions(base_dir, subject, session_list, task, contrast, space, hemisphere):
    file_paths = [
        f"{base_dir}/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_task-{task}_dir-ffx_space-{space}_hemi-{hemisphere}_ZMap-{contrast}.gii"
        for session in session_list
    ]
    
    data_sessions = [nib.load(fp).darrays[0].data for fp in file_paths]
    average_data = np.mean(data_sessions, axis=0)
    
    return average_data

def extract_fpn_activations(data, parcel_mapping):
    fpn_parcel_activations = {}
    
    for vertex, parcel in parcel_mapping.items():
        parcel_name = parcel.decode('utf-8')
        if parcel_name not in fpn_parcel_activations:
            fpn_parcel_activations[parcel_name] = []
        fpn_parcel_activations[parcel_name].append(data[:, vertex])
    
    for parcel_name in fpn_parcel_activations:
        fpn_parcel_activations[parcel_name] = np.array(fpn_parcel_activations[parcel_name])
    
    return fpn_parcel_activations

def calculate_correlation_matrix(activations):
    num_tasks = activations.shape[1]
    correlation_matrix = np.zeros((num_tasks, num_tasks))
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            correlation_matrix[i, j], _ = spearmanr(activations[:, i], activations[:, j])
    
    return correlation_matrix

def process_rsa(base_dir, subject, session_list, task, contrast, space, hemisphere, parcel_mapping):
    average_data = average_data_across_sessions(base_dir, subject, session_list, task, contrast, space, hemisphere)
    fpn_activations = extract_fpn_activations(average_data, parcel_mapping)
    
    correlation_matrices = {}
    for parcel_name, activations in fpn_activations.items():
        correlation_matrices[parcel_name] = calculate_correlation_matrix(activations)
    
    return correlation_matrices
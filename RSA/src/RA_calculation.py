import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import bct
import scipy.stats as stats
import json

def compute_ra(rsm1, rsm2):
    """Compute the regional alignment (RA) between two RSMs using cosine similarity."""
    assert rsm1.shape == rsm2.shape, "RSMs must have the same shape to compute RA."
    
    # Get the upper triangle indices, excluding the diagonal
    triu_indices = np.triu_indices(rsm1.shape[0], k=1)
    
    # Extract the upper triangle values
    rsm1_upper = rsm1[triu_indices]
    rsm2_upper = rsm2[triu_indices]
    
    # Compute cosine similarity
    ra = cosine_similarity(rsm1_upper.reshape(1, -1), rsm2_upper.reshape(1, -1))[0, 0]
    return ra

def load_rsm(file_path):
    """Load an RSM from a CSV file."""
    return np.loadtxt(file_path, delimiter=",")

def normalize_rsm(rsm):
    """Normalize an RSM by the geometric mean of its diagonal values."""
    diag = np.diag(rsm, k=0)
    if np.any(diag == 0):
        print("Warning: Zero values found in the diagonal. Skipping normalization.")
        return rsm
    geomean = stats.gmean(diag)
    normalized_rsm = np.divide(rsm, geomean)
    return normalized_rsm

def convert_to_rdm(rsm):
    """Convert an RSM to an RDM by subtracting the similarity values from 1."""
    return 1 - rsm

def main(threshold=False, save_individual=True, save_big_matrix=True, correct_geomean=True, save_type='both'):
    RSA_dir = '/home/hmueller2/ibc_code/ibc_output_RSA_cosine'
    output_dir = '/home/hmueller2/ibc_code/ibc_output_RA'

    # Create separate output directories for thresholded and raw RA matrices
    if threshold:
        output_dir = os.path.join(output_dir, 'thresholded')
    else:
        output_dir = os.path.join(output_dir, 'raw')

    # Create the topographic alignment output directory
    topographic_alignment_dir = os.path.join(output_dir, 'topographic_alignment')
    os.makedirs(topographic_alignment_dir, exist_ok=True)

    # Create subdirectories for RSM and RDM
    rsm_dir = os.path.join(topographic_alignment_dir, 'rsm')
    rdm_dir = os.path.join(topographic_alignment_dir, 'rdm')
    os.makedirs(rsm_dir, exist_ok=True)
    os.makedirs(rdm_dir, exist_ok=True)

    subjects = [d for d in os.listdir(RSA_dir) if os.path.isdir(os.path.join(RSA_dir, d)) and d.startswith('sub-')]
    
    for subject in subjects:
        subject_RSA_dir = os.path.join(RSA_dir, subject)
        subject_output_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)

        rsm_files = [f for f in os.listdir(subject_RSA_dir) if f.startswith('rsm_') and f.endswith('.csv')]
        
        ra_matrices = {}
        parcel_names = []
        
        for i, rsm_file1 in enumerate(rsm_files):
            rsm1 = load_rsm(os.path.join(subject_RSA_dir, rsm_file1))
            if correct_geomean:
                rsm1 = normalize_rsm(rsm1)
            parcel_name1 = '_'.join(rsm_file1.split('_')[1:3])  # Correctly extract the parcel name
            if parcel_name1 not in parcel_names:
                parcel_names.append(parcel_name1)
            
            for j, rsm_file2 in enumerate(rsm_files):
                if i >= j:
                    continue
                rsm2 = load_rsm(os.path.join(subject_RSA_dir, rsm_file2))
                if correct_geomean:
                    rsm2 = normalize_rsm(rsm2)
                parcel_name2 = '_'.join(rsm_file2.split('_')[1:3])  # Correctly extract the parcel name
                if parcel_name2 not in parcel_names:
                    parcel_names.append(parcel_name2)
                
                ra = compute_ra(rsm1, rsm2)
                
                if parcel_name1 not in ra_matrices:
                    ra_matrices[parcel_name1] = {}
                ra_matrices[parcel_name1][parcel_name2] = ra
        
        # Debugging: Print parcel names and number of conditions
        print(f"Subject: {subject}")
        if ra_matrices:
            first_parcel = list(ra_matrices.keys())[0]
            first_condition = list(ra_matrices[first_parcel].keys())[0]
        else:
            print("No RA matrices found.")
            continue
        
        # Save individual RA matrices to files
        if save_individual:
            for parcel1, parcel_dict in ra_matrices.items():
                for parcel2, ra_value in parcel_dict.items():
                    if threshold:
                        ra_value = bct.threshold_proportional(ra_value, 0.2)
                        suffix = '_thresholded.npy'
                    else:
                        suffix = '_raw.npy'
                    
                    ra_output_file = os.path.join(subject_output_dir, f'ra_{parcel1}_vs_{parcel2}_{subject}{suffix}')
                    np.save(ra_output_file, ra_value)
        
        # Initialize the big matrix
        if save_big_matrix:
            n_parcels = len(parcel_names)
            big_matrix = np.zeros((n_parcels, n_parcels))
            
            # Fill the big matrix with RA values
            for i, parcel1 in enumerate(parcel_names):
                for j, parcel2 in enumerate(parcel_names):
                    if parcel1 in ra_matrices and parcel2 in ra_matrices[parcel1]:
                        ra_value = ra_matrices[parcel1][parcel2]
                        big_matrix[i, j] = ra_value
                    elif parcel2 in ra_matrices and parcel1 in ra_matrices[parcel2]:
                        ra_value = ra_matrices[parcel2][parcel1]
                        big_matrix[i, j] = ra_value
            
            # Set the diagonal to 1 (self-similarity)
            np.fill_diagonal(big_matrix, 1)

            # Convert the big matrix to an RDM
            big_matrix_rdm = convert_to_rdm(big_matrix)
            
            # Save the big matrix to a file based on save_type
            if save_type in ['both', 'rsm']:
                big_matrix_output_file_rsm = os.path.join(rsm_dir, f'topographic_alignment_rsm_{subject}.npy')
                np.save(big_matrix_output_file_rsm, big_matrix)
            if save_type in ['both', 'rdm']:
                big_matrix_output_file_rdm = os.path.join(rdm_dir, f'topographic_alignment_rdm_{subject}.npy')
                np.save(big_matrix_output_file_rdm, big_matrix_rdm)
            
            # Save the parcel names
            parcel_names_file = os.path.join(topographic_alignment_dir, f'parcel_names_{subject}.json')
            with open(parcel_names_file, 'w') as f:
                json.dump(parcel_names, f)
        
        print(f"--- Done with subject {subject} ---")
    print(f"All topographic alignment matrices have been saved in {topographic_alignment_dir}.")

if __name__ == "__main__":
    # Set threshold, save_individual, save_big_matrix, and save_type based on your requirement
    main(threshold=False, save_individual=True, save_big_matrix=True, correct_geomean=True, save_type='both')
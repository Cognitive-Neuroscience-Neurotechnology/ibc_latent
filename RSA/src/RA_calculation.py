import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import bct

def compute_ra(rsm1, rsm2):
    """Compute the regional alignment (RA) between two RSMs using cosine similarity."""
    assert rsm1.shape == rsm2.shape, "RSMs must have the same shape to compute RA."
    ra = cosine_similarity(rsm1, rsm2)
    return ra

def load_rsm(file_path):
    """Load an RSM from a CSV file."""
    return np.loadtxt(file_path, delimiter=",")

def main(threshold=False, save_individual=True, save_big_matrix=True):
    RSA_dir = '/home/hmueller2/ibc_code/ibc_output_RSA_cosine'
    output_dir = '/home/hmueller2/ibc_code/ibc_output_RA_npy'

    # Create separate output directories for thresholded and raw RA matrices
    if threshold:
        output_dir = os.path.join(output_dir, 'thresholded')
    else:
        output_dir = os.path.join(output_dir, 'raw')

    # Create the topographic alignment output directory
    topographic_alignment_dir = os.path.join(output_dir, 'topographic_alignment')
    os.makedirs(topographic_alignment_dir, exist_ok=True)

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
            parcel_name1 = '_'.join(rsm_file1.split('_')[1:3])  # Correctly extract the parcel name
            if parcel_name1 not in parcel_names:
                parcel_names.append(parcel_name1)
            
            for j, rsm_file2 in enumerate(rsm_files):
                if i >= j:
                    continue
                rsm2 = load_rsm(os.path.join(subject_RSA_dir, rsm_file2))
                parcel_name2 = '_'.join(rsm_file2.split('_')[1:3])  # Correctly extract the parcel name
                if parcel_name2 not in parcel_names:
                    parcel_names.append(parcel_name2)
                
                ra = compute_ra(rsm1, rsm2)
                
                if parcel_name1 not in ra_matrices:
                    ra_matrices[parcel_name1] = {}
                ra_matrices[parcel_name1][parcel_name2] = ra
        
        # Save individual RA matrices to files
        if save_individual:
            for parcel1, parcel_dict in ra_matrices.items():
                for parcel2, ra_matrix in parcel_dict.items():
                    if threshold:
                        ra_matrix = bct.threshold_proportional(ra_matrix, 0.2)
                        suffix = '_thresholded.npy'
                    else:
                        suffix = '_raw.npy'
                    
                    ra_output_file = os.path.join(subject_output_dir, f'ra_{parcel1}_vs_{parcel2}_{subject}{suffix}')
                    np.save(ra_output_file, ra_matrix)
        
        # Initialize the big matrix
        if save_big_matrix:
            n_parcels = len(parcel_names)
            big_matrix = np.zeros((n_parcels, n_parcels))
            
            # Fill the big matrix
            for i, parcel1 in enumerate(parcel_names):
                for j, parcel2 in enumerate(parcel_names):
                    if parcel1 in ra_matrices and parcel2 in ra_matrices[parcel1]:
                        big_matrix[i, j] = np.mean(ra_matrices[parcel1][parcel2])
                    elif parcel2 in ra_matrices and parcel1 in ra_matrices[parcel2]:
                        big_matrix[i, j] = np.mean(ra_matrices[parcel2][parcel1])
            
            # Save the big matrix to a file
            big_matrix_output_file = os.path.join(topographic_alignment_dir, f'topographic_alignment_{subject}.npy')
            np.save(big_matrix_output_file, big_matrix)
        
        print(f"Done with subject {subject}")

if __name__ == "__main__":
    # Set threshold, save_individual, and save_big_matrix based on your requirement
    main(threshold=False, save_individual=False, save_big_matrix=True)
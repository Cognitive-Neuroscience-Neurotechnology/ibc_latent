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

def main(threshold=False):
    RSA_dir = '/home/hmueller2/ibc_code/ibc_output_RSA_cosine'
    output_dir = '/home/hmueller2/ibc_code/ibc_output_RA_npy'

    # Create separate output directories for thresholded and raw RA matrices
    if threshold:
        output_dir = os.path.join(output_dir, 'thresholded')
    else:
        output_dir = os.path.join(output_dir, 'raw')

    subjects = [d for d in os.listdir(RSA_dir) if os.path.isdir(os.path.join(RSA_dir, d)) and d.startswith('sub-')]
    
    for subject in subjects:
        subject_RSA_dir = os.path.join(RSA_dir, subject)
        subject_output_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)

        rsm_files = [f for f in os.listdir(subject_RSA_dir) if f.startswith('rsm_') and f.endswith('.csv')]
        
        ra_matrices = {}
        
        for i, rsm_file1 in enumerate(rsm_files):
            rsm1 = load_rsm(os.path.join(subject_RSA_dir, rsm_file1))
            parcel_name1 = '_'.join(rsm_file1.split('_')[1:3])  # Correctly extract the parcel name
            
            for j, rsm_file2 in enumerate(rsm_files):
                if i >= j:
                    continue
                rsm2 = load_rsm(os.path.join(subject_RSA_dir, rsm_file2))
                parcel_name2 = '_'.join(rsm_file2.split('_')[1:3])  # Correctly extract the parcel name
                
                ra = compute_ra(rsm1, rsm2)
                
                if parcel_name1 not in ra_matrices:
                    ra_matrices[parcel_name1] = {}
                ra_matrices[parcel_name1][parcel_name2] = ra
        
        # Save RA matrices to files
        for parcel1, parcel_dict in ra_matrices.items():
            for parcel2, ra_matrix in parcel_dict.items():
                if threshold:
                    ra_matrix = bct.threshold_proportional(ra_matrix, 0.2)
                    suffix = '_thresholded.npy'
                else:
                    suffix = '_raw.npy'
                
                ra_output_file = os.path.join(subject_output_dir, f'ra_{parcel1}_vs_{parcel2}_{subject}{suffix}')
                np.save(ra_output_file, ra_matrix)
        print(f"Done with subject {subject}")

if __name__ == "__main__":
    # Set threshold to True or False based on your requirement
    main(threshold=False)
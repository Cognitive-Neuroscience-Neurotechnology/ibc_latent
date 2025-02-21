import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_ra(rsm1, rsm2):
    """Compute the regional alignment (RA) between two RSMs using cosine similarity."""
    assert rsm1.shape == rsm2.shape, "RSMs must have the same shape to compute RA."
    rsm1_flat = rsm1.flatten().reshape(1, -1)
    rsm2_flat = rsm2.flatten().reshape(1, -1)
    ra = cosine_similarity(rsm1_flat, rsm2_flat)[0, 0]
    return ra

def load_rsm(file_path):
    """Load an RSM from a CSV file."""
    return np.loadtxt(file_path, delimiter=",")

def main():
    RSA_dir = '/home/hmueller2/ibc_code/ibc_output_RSA_cosine'
    output_dir = '/home/hmueller2/ibc_code/ibc_output_RA'

    subjects = [d for d in os.listdir(RSA_dir) if os.path.isdir(os.path.join(RSA_dir, d)) and d.startswith('sub-')]
    
    for subject in subjects:
        subject_RSA_dir = os.path.join(RSA_dir, subject)
        subject_output_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)
        ra_results = []

        rsm_files = [f for f in os.listdir(subject_RSA_dir) if f.startswith('rsm_') and f.endswith('.csv')]
        
        for i, rsm_file1 in enumerate(rsm_files):
            rsm1 = load_rsm(os.path.join(subject_RSA_dir, rsm_file1))
            parcel_name1 = rsm_file1.split('_')[1]
            
            for j, rsm_file2 in enumerate(rsm_files):
                if i >= j:
                    continue
                rsm2 = load_rsm(os.path.join(subject_RSA_dir, rsm_file2))
                parcel_name2 = rsm_file2.split('_')[1]
                
                ra = compute_ra(rsm1, rsm2)
                ra_results.append([subject, parcel_name1, parcel_name2, ra])
        
        # Save RA results to a CSV file
        ra_df = pd.DataFrame(ra_results, columns=['subject', 'parcel1', 'parcel2', 'ra'])
        ra_output_file = os.path.join(subject_output_dir, f'ra_sub-{subject}.csv')
        ra_df.to_csv(ra_output_file, index=False)
        print(f"RA results saved to {ra_output_file}")

if __name__ == "__main__":
    main()
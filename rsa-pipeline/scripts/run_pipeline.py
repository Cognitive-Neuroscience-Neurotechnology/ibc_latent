import os
import numpy as np
import pandas as pd
import nibabel as nib
from src.load_data import load_surface_maps
from src.process_data import process_rsa
from src.utils import get_subjects, get_sessions, get_contrasts, get_hemispheres

def run_pipeline():
    base_dir = '/home/hmueller2/Downloads/contrast_maps/resulting_smooth_maps_surface/'
    output_dir = '/home/hmueller2/ibc_code/ibc_output_RSA'
    
    subjects = get_subjects()
    sessions = get_sessions()
    contrasts = get_contrasts()
    hemispheres = get_hemispheres()

    for subject in subjects:
        for session in sessions:
            for contrast in contrasts:
                for hemisphere in hemispheres:
                    print(f"Processing subject: {subject}, session: {session}, contrast: {contrast}, hemisphere: {hemisphere}")
                    
                    # Load the data
                    file_paths = [f"{base_dir}/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_task-*_dir-ffx_space-fsaverage7_hemi-{hemisphere}_ZMap-{contrast}.gii"]
                    data = load_surface_maps(file_paths)

                    # Process the data
                    results = process_rsa(data, subject, session, contrast, hemisphere)

                    # Save the results
                    output_file = os.path.join(output_dir, f"results_sub-{subject}_ses-{session}_task-*_contrast-{contrast}_hemi-{hemisphere}.npy")
                    np.save(output_file, results)

if __name__ == "__main__":
    run_pipeline()
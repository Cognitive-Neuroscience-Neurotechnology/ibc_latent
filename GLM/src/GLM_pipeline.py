# RUN config.py BEFORE RUNNING THIS SCRIPT

import os
import glob
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn import plotting, image, glm
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.plotting import plot_design_matrix, plot_stat_map
import matplotlib.pyplot as plt

# Import from config and task_contrasts
from config import base_dir, output_dir
from RSA.task_contrasts import task_contrasts

# Define the parameters to iterate over
subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')] # Extract the subject ID after 'sub-'
tasks = task_contrasts.keys() # Taken from task_contrasts.py
directions = ['ap', 'pa']

# Ensure the output directory exists
if not os.path.exists(output_dir):
    raise FileNotFoundError(f"ERROR: The output directory {output_dir} does not exist.")
else:
    print(f"Output directory: {output_dir}")

def find_fmri_files(base_dir, subject, task):
    search_pattern = f"sub-{subject}_ses-*_task-{task}_dir-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    full_search_pattern = os.path.join(base_dir, '**', search_pattern)
    fmri_files = glob.glob(full_search_pattern, recursive=True)
    print(f"  Found {len(fmri_files)} fMRI files for subject {subject}, task {task}")
    return fmri_files

def build_and_save_pipeline(subject, task, session, direction, run=None):
    print(f"Processing subject: {subject}, task: {task}, session: {session}, direction: {direction}, run: {run}")
    
    events_file_pattern = f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_events.tsv'
    if run:
        events_file_pattern = f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_run-{run}_events.tsv'
    events_file_path = glob.glob(os.path.join(base_dir, f'sub-{subject}', f'ses-{session}', 'func', events_file_pattern), recursive=True)[0]
    events_data = pd.read_csv(events_file_path, sep='\t')
    conditions = events_data['trial_type'].unique()
    print(f"  Conditions: {conditions}")

    fmri_file_pattern = f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    if run:
        fmri_file_pattern = f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    fmri_file_path = glob.glob(os.path.join(base_dir, f'sub-{subject}', f'ses-{session}', 'func', fmri_file_pattern), recursive=True)[0]
    fmri_img = nib.load(fmri_file_path)
    print(f"-- Loaded fMRI file: {fmri_file_path}")

    time_series_file_pattern = f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_desc-confounds_timeseries.tsv'
    if run:
        time_series_file_pattern = f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_run-{run}_desc-confounds_timeseries.tsv'
    time_series_file_path = glob.glob(os.path.join(base_dir, f'sub-{subject}', f'ses-{session}', 'func', time_series_file_pattern), recursive=True)[0]
    time_series_data = pd.read_csv(time_series_file_path, sep='\t')
    print(f"-- Loaded confounds file: {time_series_file_path}")

    tr = 2.0
    n_scans = fmri_img.shape[-1]
    frame_times = np.arange(n_scans) * tr

    # Create the design matrix with the confounds included
    confounds = time_series_data[['tx', 'ty', 'tz', 'rx', 'ry', 'rz']]
    design_matrix = make_first_level_design_matrix(frame_times, events_data[events_data['trial_type'].isin(conditions)], hrf_model='spm', add_regs=confounds)

    # Save the design matrix
    output_task_dir = os.path.join(output_dir, f'sub-{subject}', task, f'ses-{session}', f'dir-{direction}')
    os.makedirs(output_task_dir, exist_ok=True)
    design_matrix_file_path = os.path.join(output_task_dir, f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_design_matrix.csv')
    design_matrix.to_csv(design_matrix_file_path)
    print(f"Saved design matrix to {design_matrix_file_path}")

    glm_model = FirstLevelModel(t_r=tr, noise_model='ar1', standardize=True)
    glm_model = glm_model.fit(fmri_img, design_matrices=design_matrix)
    
    # Extract beta maps for each contrast
    for condition in conditions:
        contrast_name = condition.replace(' ', '_').replace('-', '_vs_')
        print(f"  Processing contrast: {contrast_name}")
        contrast_map = glm_model.compute_contrast(condition, output_type='effect_size')
        if contrast_map is not None:
            if run:
                output_file_path = os.path.join(output_task_dir, f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_run-{run}_contrast-{contrast_name}.nii.gz')
            else:
                output_file_path = os.path.join(output_task_dir, f'sub-{subject}_ses-{session}_task-{task}_dir-{direction}_contrast-{contrast_name}.nii.gz')
            contrast_map.to_filename(output_file_path)
            print(f"Saved contrast map to {output_file_path}")
        else:
            print(f"ERROR: No contrast map generated for contrast: {contrast_name}")

# Iterate through all parameter combinations
for subject in subjects:
    print("\n" + "-" * 80 + "\n")
    print(f"Subject: {subject}")
    for task in tasks:
        print("\n" + "-" * 80 + "\n")
        print(f"Task: {task}")
        fmri_files = find_fmri_files(base_dir, subject, task)
        if not fmri_files:
            print(f"  ERROR: No fMRI files found for subject {subject}, task {task}")
            continue
        sessions = set()
        for f in fmri_files:
            try:
                session = os.path.basename(f).split('_')[1].split('-')[1]
                sessions.add(session)
            except IndexError:
                print(f"  ERROR: Unexpected file format for {f}")
                continue
        for session in sessions:
            print("-" * 80)
            print(f"    Session: {session}")
            for direction in directions:
                # Process files with and without run information separately
                run_files = [f for f in fmri_files if '_run-' in f and f'_dir-{direction}_' in f]
                non_run_files = [f for f in fmri_files if '_run-' not in f and f'_dir-{direction}_' in f]
                
                if run_files:
                    for run_file in run_files:
                        try:
                            print(f"Processing file: {run_file}")  # Debugging statement
                            run_parts = [part for part in os.path.basename(run_file).split('_') if part.startswith('run-')]
                            print(f"Run parts: {run_parts}")  # Debugging statement
                            if run_parts:
                                run = run_parts[0].split('-')[1]
                                print(f"      Processing run: {run}")
                                build_and_save_pipeline(subject, task, session, direction, run)
                            else:
                                raise IndexError
                        except IndexError:
                            print(f"  ERROR: Unexpected file format for {run_file}")
                            continue
                
                if non_run_files:
                    print(f"      Processing without run")
                    build_and_save_pipeline(subject, task, session, direction)
    print(f"SUBJECT {subject} IS DONE :)")

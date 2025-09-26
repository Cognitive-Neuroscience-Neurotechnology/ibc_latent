import os
import pandas as pd
import glob
import nibabel as nib

# What the following script does: 
# 1) For each subject, it calculates the total duration of all tasks from the event files in the func/ folders.
# 2) It counts the total number of frames in all cleaned .dtseries.nii

# Paths
base_dir = "/ptmp/hmueller2/Downloads/ibc_preprocessed"
glm_base = "/ptmp/hmueller2/Downloads/fmriprep_out"
overview_path = "/home/hmueller2/ibc_code/ibc_latent/Data Info/session_overview_unscrubbed.csv"
subjects_file = "/ptmp/hmueller2/Downloads/subjects_resting.txt"

# Load subject list
with open(subjects_file) as f:
    subject_list = [line.strip().zfill(2) for line in f if line.strip()]

# Load overview CSV
if not os.path.exists(overview_path):
    # Create a new DataFrame with just the subject column
    overview_df = pd.DataFrame({'subject': [int(s) for s in subject_list]})
else:
    overview_df = pd.read_csv(overview_path)

overview_df["overall_task_minutes"] = 0.0  # Initialize column as float for minutes
overview_df["total_cleaned_frames"] = 0    # New column for total frames
overview_df["total_unscrubbed_frames"] = 0    # New column for unscrubbed frames

for subject_id in subject_list:
    total_duration = 0.0
    total_frames = 0
    total_unscrubbed_frames = 0

    # Find all func/ folders within this subject directory (func = unscrubbed)
    subject_folder = os.path.join(base_dir, f"sub-{subject_id}")
    for root, dirs, files in os.walk(subject_folder):
        if os.path.basename(root) == "func":
            for file in files:
                if file.endswith("_dir-ap_events.tsv"):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, sep="\t")
                        if "duration" in df.columns:
                            total_duration += df["duration"].sum()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # Check all cleaned files for this subject
    glm_dirs = glob.glob(os.path.join(glm_base, f"sub-{subject_id}", "ses-*", "postfmriprep_scrubbed", "GLM"))
    for glm_dir in glm_dirs:
        cleaned_files = glob.glob(os.path.join(glm_dir, f"sub-{subject_id}_ses-*_task*_dir-*_cleaned.dtseries.nii"))
        for cleaned_file in cleaned_files:
            try:
                img = nib.load(cleaned_file)
                total_frames += img.shape[0]
            except Exception as e:
                print(f"Error reading {cleaned_file}: {e}")

    # Find all demeaned/detrended files for this subject
    demean_dirs = glob.glob(os.path.join(glm_base, f"sub-{subject_id}", "ses-*", "postfmriprep", "demean"))
    for demean_dir in demean_dirs:
        nifti_files = glob.glob(os.path.join(demean_dir, f"sub-{subject_id}_ses-*_task-*_dir-*_demean_detrend.dtseries.nii"))
        for nifti_file in nifti_files:
            try:
                img = nib.load(nifti_file)
                total_unscrubbed_frames += img.shape[0]
            except Exception as e:
                print(f"Error reading {nifti_file}: {e}")

    # Update overview
    total_duration_minutes = round(total_duration / 60, 1)
    overview_df.loc[overview_df["subject"] == int(subject_id), "overall_task_minutes"] = total_duration_minutes
    overview_df.loc[overview_df["subject"] == int(subject_id), "total_cleaned_frames"] = total_frames
    overview_df.loc[overview_df["subject"] == int(subject_id), "total_unscrubbed_frames"] = total_unscrubbed_frames

# Save updated CSV
overview_df.to_csv(overview_path, index=False)

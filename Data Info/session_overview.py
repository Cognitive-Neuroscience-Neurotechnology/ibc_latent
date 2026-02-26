import os
import pandas as pd
import glob

# Paths
base_dir = "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/ibc_preprocessed_MNI"
overview_path = "/home/hmueller2/ibc_code/ibc_latent/Data Info/session_overview.csv"

# Load overview CSV
overview_df = pd.read_csv(overview_path)
overview_df["overall_task_minutes"] = 0.0  # Initialize column as float for minutes

# Go through all subject folders
for subject_folder in glob.glob(os.path.join(base_dir, "sub-*")):
    subject_id = os.path.basename(subject_folder).replace("sub-", "")
    total_duration = 0.0

    # Find all func/ folders within this subject directory (func = unscrubbed)
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

    # Convert total duration to minutes, round to 1 decimal place, and fill in overall_task_minutes for this subject
    total_duration_minutes = round(total_duration / 60, 1)
    overview_df.loc[overview_df["subject"] == int(subject_id), "overall_task_minutes"] = total_duration_minutes

# Save updated CSV
overview_df.to_csv(overview_path, index=False)

import os
import pandas as pd

# Paths
csv_path = "ptmp/hmueller2/ibc_code/ibc_latent/Data Info/subject_contrast_counts.csv"
base_path = "ptmp/hmueller2/Downloads/ibc_preprocessed"

# Load original CSV
df = pd.read_csv(csv_path)

# Initialize new columns
df["session_count"] = 0
df["overall_time"] = 0  # Placeholder

# Iterate over each subject to get session count
for idx, row in df.iterrows():
    subject = row["subject"]
    subject_dir = os.path.join(base_path, f"sub-{subject}")
    try:
        sessions = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
        df.at[idx, "session_count"] = len(sessions)
    except FileNotFoundError:
        print(f"Directory not found for subject: {subject}")
        df.at[idx, "session_count"] = 0

# Save updated CSV
df.to_csv(csv_path, index=False)

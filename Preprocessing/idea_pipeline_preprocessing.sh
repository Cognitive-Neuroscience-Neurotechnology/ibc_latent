#!/bin/bash

# Usage: ./post_fmriprep_pipeline.sh <subject> <session> <task> <direction>
# Example: ./post_fmriprep_pipeline.sh 01 01 rest LR

subject=$1
session=$2
task=$3
direction=$4

# Set your base directory
base_dir="/ptmp/hmueller2/Downloads/fmriprep_out"
func_dir="${base_dir}/sub-${subject}/ses-${session}/func"
# Output directory
out_dir="${base_dir}/sub-${subject}/ses-${session}/postfmriprep"
mkdir -p "${out_dir}"

# File naming
# Example file: sub-01_ses-15_task-RestingState_dir-ap_space-fsLR_den-91k_bold.dtseries.nii
fBaseName="sub-${subject}_ses-${session}_task-${task}_dir-${direction}"
bold="${func_dir}/${fBaseName}_space-fsLR_den-91k_bold.dtseries.nii"
confounds="${func_dir}/${fBaseName}_desc-confounds_timeseries.tsv"
json="${func_dir}/${fBaseName}_space-fsLR_den-91k_bold.json"

# Output directory
out_dir="${base_dir}/sub-${subject}/ses-${session}/postfmriprep"
mkdir -p "${out_dir}"

# 1. Extract FD and apply threshold (scrubbing mask)
echo "***** 1: Calculating Framewise Displacement and Scrubbing Mask *****"
FD_mask="${out_dir}/${fBaseName}_FD_scrub_mask.txt"
FD_threshold=0.2
python3 Dependencies/Aradia/process_fd.py -i "${confounds}" -FD ${FD_threshold} -o "${FD_mask}"


FD_col=$(head -1 ${confounds} | tr '\t' '\n' | grep -n '^framewise_displacement$' | cut -d: -f1)
awk -v col=${FD_col} -v thresh=0.2 'NR>1 {print ($col=="" ? 0 : ($col>thresh ? 1 : 0))}' ${confounds} > "${out_dir}/${fBaseName}_FD_scrub_mask.txt"

# 2. Demean and detrend BOLD. Note, 3dDetrend (AFNI) does not support cifti files directly.
# Convert dtseries.nii to nifti if needed, or use .nii.gz directly
# detrended="${out_dir}/${fBaseName}_detrend.nii.gz"
# 3dDetrend -prefix "${detrended}" -polort 1 "${bold}"
detrended="${out_dir}/${fBaseName}_detrend.dtseries.nii"
python Dependencies/detrend_cifti.py "${bold}" "${detrended}"

# 3. Remove initial volumes if needed (e.g., first 5). Note, fslval and fslroi do not support CIFTI files.
# n_remove=5
# n_vols=$(fslval "${detrended}" dim4)
# n_keep=$((n_vols - n_remove))
# detrended_trim="${out_dir}/${fBaseName}_detrend_trim.nii.gz"
# fslroi "${detrended}" "${detrended_trim}" ${n_remove} ${n_keep}
n_remove=5
detrended_trim="${out_dir}/${fBaseName}_detrend_trim.dtseries.nii"
python Dependencies/trim_cifti.py "${detrended}" "${detrended_trim}" ${n_remove}

# 4. Prepare regressors (motion, WM, CSF, global) from confounds. PYTHON.
# Example: extract columns and remove first 5 rows
python <<EOF
import pandas as pd
conf = pd.read_csv("${confounds}", sep='\t')
cols = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','white_matter','csf','global_signal']
conf[cols].iloc[${n_remove}:].to_csv("${out_dir}/${fBaseName}_regressors.txt", sep='\t', index=False, header=False)
EOF

# 5. (Optional) Z-score regressors. PYTHON.
python <<EOF
import numpy as np
reg = np.loadtxt("${out_dir}/${fBaseName}_regressors.txt")
reg_z = (reg - reg.mean(axis=0)) / reg.std(axis=0)
np.savetxt("${out_dir}/${fBaseName}_regressors_z.txt", reg_z, fmt="%.6f")
EOF

# 6. Nuisance regression, scrubbing, bandpass filtering (use Nilearn, AFNI, or custom Python)
# Example: Nilearn or custom script (not shown here)

echo "Pipeline complete. Outputs in ${out_dir}"
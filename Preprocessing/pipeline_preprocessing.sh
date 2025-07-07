#!/bin/bash

# Usage: ./pipeline_preprocessing.sh <subject> <session> <task> <direction>
# Example: ./pipeline_preprocessing.sh 01 01 rest LR

set -e

subject=$1
session=$2
task=$3
direction=$4

base_dir="/ptmp/hmueller2/Downloads/fmriprep_out"
func_dir="${base_dir}/sub-${subject}/ses-${session}/func"
out_dir="${base_dir}/sub-${subject}/ses-${session}/postfmriprep"
mkdir -p "${out_dir}"

fBaseName="sub-${subject}_ses-${session}_task-${task}_dir-${direction}"
bold="${func_dir}/${fBaseName}_space-fsLR_den-91k_bold.dtseries.nii"
confounds="${func_dir}/${fBaseName}_desc-confounds_timeseries.tsv"
json="${func_dir}/${fBaseName}_space-fsLR_den-91k_bold.json"

echo "***** 1: Calculating Framewise Displacement and Scrubbing Mask *****"
FD_mask="${out_dir}/${fBaseName}_FD_scrub_mask.txt"
FD_threshold=0.2
python3 Dependencies/Aradia/process_fd.py -i "${confounds}" -FD ${FD_threshold} -o "${FD_mask}"

echo "***** 2: Demeaning and Detrending BOLD *****"
detrended="${out_dir}/${fBaseName}_detrend.dtseries.nii"
python3 Dependencies/detrend_cifti.py "${bold}" "${detrended}"

echo "***** 3: Removing Initial Volumes *****"
n_remove=5
detrended_trim="${out_dir}/${fBaseName}_detrend_trim.dtseries.nii"
python3 Dependencies/trim_cifti.py "${detrended}" "${detrended_trim}" ${n_remove}

echo "***** 4: Preparing Regressors (Motion, WM, CSF, Global) *****"
regressors_txt="${out_dir}/${fBaseName}_regressors.txt"
python3 <<EOF
import pandas as pd
conf = pd.read_csv("${confounds}", sep='\t')
cols = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','white_matter','csf','global_signal']
conf[cols].iloc[${n_remove}:].to_csv("${regressors_txt}", sep=' ', index=False, header=False)
EOF

echo "***** 5: Demeaning, Detrending, Z-scoring Regressors *****"
regressors_dm="${out_dir}/${fBaseName}_regressors_demeaned_detrended.txt"
regressors_z="${out_dir}/${fBaseName}_regressors_z.txt"
python3 Dependencies/Aradia/demean_detrend_reg.py -i "${regressors_txt}" -odmdt "${regressors_dm}" -o "${regressors_z}"

echo "***** 6: Plotting Regressors *****"
reg_png="${out_dir}/${fBaseName}_regressors_z.png"
python3 Dependencies/Aradia/regressor_plots.py -i "${regressors_z}" -glob yes -o "${reg_png}"

echo "***** 7: Regression, Scrubbing, Bandpass Filtering *****"
# Extract TR from JSON
TR=$(jq -r '.RepetitionTime' "${json}")
echo "The TR is $TR seconds."

cleaned_bold="${out_dir}/${fBaseName}_cleaned.dtseries.nii"
python3 Dependencies/Aradia/regression_interpolation.py \
    -i "${detrended_trim}" \
    -r "${regressors_z}" \
    -FD "${FD_mask}" \
    -TR "${TR}" \
    -o "${cleaned_bold}"

echo "Pipeline complete. Outputs in ${out_dir}"
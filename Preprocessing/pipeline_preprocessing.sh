#!/bin/bash

# Usage: ./pipeline_preprocessing.sh <subject> <session> <task> <direction>
# Example: ./pipeline_preprocessing.sh 01 15 RestingState pa

set -e

subject=$1
session=$2
task=$3
direction=$4

base_dir="/ptmp/hmueller2/Downloads/fmriprep_out"
func_dir="${base_dir}/sub-${subject}/ses-${session}/func"
out_dir="${base_dir}/sub-${subject}/ses-${session}/postfmriprep"
mkdir -p "${out_dir}"
demean_dir="${out_dir}/demean"
regressors_dir="${out_dir}/regressors"
glm_dir="${out_dir}/GLM"
plots_dir="${out_dir}/plots"
mkdir -p "${demean_dir}" "${regressors_dir}" "${glm_dir}" "${plots_dir}"

fBaseName="sub-${subject}_ses-${session}_task-${task}_dir-${direction}"
bold="${func_dir}/${fBaseName}_space-fsLR_den-91k_bold.dtseries.nii"
confounds="${func_dir}/${fBaseName}_desc-confounds_timeseries.tsv"
json="${func_dir}/${fBaseName}_space-fsLR_den-91k_bold.json"

# -1-: Mark timepoints with FD > 0.2 as contaminated (1), others as clean (0).
echo "***** 1: Build scrubbing mask from FD values *****"
FD_mask="${regressors_dir}/${fBaseName}_FD_scrub_mask.txt"
FD_threshold=0.2
# Different from Aradia. process_fd.py is not needed, as I already have fd values in confounds.
# Fill missing values with 0 (as fmriprep does for the first row).
python3 <<EOF
import pandas as pd
conf = pd.read_csv("${confounds}", sep='\t')
fd = conf['framewise_displacement'].fillna(0)
mask = (fd > ${FD_threshold}).astype(int)
out = pd.DataFrame({'FD': fd, 'contam': mask})
out.to_csv("${FD_mask}", sep=' ', index=False, header=False)
EOF

# -2-: Detrend and demean with workbench
echo "***** 2: Demeaning and Detrending BOLD *****"
detrended="${demean_dir}/${fBaseName}_detrend.dtseries.nii"
meaned="${demean_dir}/${fBaseName}_mean.dtseries.nii"
# I use workbench instead of fslmaths because cifti files are not supported by fslmaths.

# Calculate mean across time
wb_command -cifti-reduce "${bold}" MEAN "${meaned}"

# Subtract mean from each timepoint (demean)
wb_command -cifti-math "x - m" "${detrended}" -var x "${bold}" -var m "${meaned}"

echo "demeaned std:"
wb_command -cifti-stats "${detrended}" -reduce STDEV -axis 1

# -3-
echo "***** 3: Removing Initial Volumes *****"
# Remove initial transients / artifacts
# = Step 5 in Aradia's pipeline
# Cannot use fslroi because cifti files are not supported by fslroi.
n_remove=5
detrended_trim="${demean_dir}/${fBaseName}_detrend_trim.dtseries.nii"
python3 Dependencies/trim_cifti.py "${detrended}" "${detrended_trim}" ${n_remove}

# -4-
echo "***** 4: Preparing Regressors (Motion, WM, CSF, Global) *****"
regressors_txt="${regressors_dir}/${fBaseName}_regressors.txt"
python3 <<EOF
import pandas as pd
conf = pd.read_csv("${confounds}", sep='\t')
cols = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'white_matter', 'white_matter_derivative1',
    'csf', 'csf_derivative1',
    'global_signal', 'global_signal_derivative1'
]
conf[cols].iloc[${n_remove}:].to_csv("${regressors_txt}", sep=' ', index=False, header=False)
EOF

# -5- (Same as Aradia's step 5b)
echo "***** 5: Demeaning, Detrending, Z-scoring Regressors *****"
regressors_dm="${regressors_dir}/${fBaseName}_regressors_demeaned_detrended.txt"
regressors_z="${regressors_dir}/${fBaseName}_regressors_z.txt"
python3 Aradia/demean_detrend_reg.py -i "${regressors_txt}" -odmdt "${regressors_dm}" -o "${regressors_z}"

# -6- (Same as Aradia's step 5c)
echo "***** 6: Plotting Regressors *****"
reg_png="${plots_dir}/${fBaseName}_regressors_z.png"
python3 Aradia/regressor_plots.py -i "${regressors_z}" -glob yes -o "${reg_png}"

# -7- 
echo "***** 7: Regression, Scrubbing, Bandpass Filtering *****"
# Extract TR from JSON
TR=$(jq -r '.RepetitionTime' "${json}")
echo "The TR is $TR seconds."

cleaned_bold="${glm_dir}/${fBaseName}_cleaned.dtseries.nii"
python3 Aradia/regression_interpolation.py \
    -i "${detrended_trim}" \
    -r "${regressors_z}" \
    -FD "${FD_mask}" \
    -TR "${TR}" \
    -o "${cleaned_bold}"


echo "***** Finalizing Outputs *****"


echo "Pipeline complete. Outputs in ${out_dir}"
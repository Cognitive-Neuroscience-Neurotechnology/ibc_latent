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
medial_wall_dir="/ptmp/hmueller2/Downloads/fsLR_masks"

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
demeaned_detrended="${demean_dir}/${fBaseName}_demean_detrend.dtseries.nii"
python3 <<EOF
import nibabel as nib
import numpy as np
from scipy.signal import detrend

img = nib.load("${bold}")
data = img.get_fdata()
# Demean and detrend along time axis (last axis)
demeaned = data - data.mean(axis=-1, keepdims=True)
detrended = detrend(demeaned, axis=-1, type='linear')
nib.save(nib.Cifti2Image(detrended, img.header), "${demeaned_detrended}")
EOF

detrended="${demeaned_detrended}"

echo "demeaned std:"
python3 <<EOF
import nibabel as nib
import numpy as np
img = nib.load("${detrended}")
data = img.get_fdata()
print(np.std(data, axis=-1))
EOF


# -3- Medial wall removal (keep subcortex) using RR_utils
echo "***** 3: Medial wall removal (keep subcortex) *****"

detrended_nomw="${demean_dir}/${fBaseName}_detrend_nomw.dtseries.nii"

python3 <<EOF
import os
import subprocess

dtseries = "${detrended}"
ciftify_dir = "${medial_wall_dir}"
resolution = "32k"

mw_dir = os.path.join(ciftify_dir)
left_mw = os.path.join(mw_dir, "L.atlasroi.32k_fs_LR.shape.gii")
right_mw = os.path.join(mw_dir, "R.atlasroi.32k_fs_LR.shape.gii")

print("dtseries:", dtseries)

if dtseries.endswith('.dtseries.nii'):
    output_file = dtseries.replace('.dtseries.nii', '_no_medial_wall.dtseries.nii')
elif dtseries.endswith('.dlabel.nii'):
    output_file = dtseries.replace('.dlabel.nii', '_no_medial_wall.dlabel.nii')
elif dtseries.endswith('.dscalar.nii'):
    output_file = dtseries.replace('.dscalar.nii', '_no_medial_wall.dscalar.nii')
    print('Code is not optimized for this file type. Attempting medial wall removal.')
else:
    raise ValueError("Input file must be .dtseries.nii or .dlabel.nii")

subprocess.run([
    'wb_command', '-cifti-restrict-dense-map',
    dtseries,
    'COLUMN',
    output_file,
    '-left-roi', left_mw,
    '-right-roi', right_mw
], check=True)

# Move to expected output location
import shutil
shutil.move(output_file, "${detrended_nomw}")
EOF

# Use the new file for downstream steps
detrended="${detrended_nomw}"
echo "***** Output file structure (R, L, subcortex) *****"
python3 <<EOF
import nibabel as nib
img = nib.load("${detrended}")
axes = img.header.get_axis(1)
if hasattr(axes, 'brain_models'):
    bm = axes.brain_models
    left = sum(bm.model[0].index_count for bm in bm if bm.model[0].structure == 'CIFTI_STRUCTURE_CORTEX_LEFT')
    right = sum(bm.model[0].index_count for bm in bm if bm.model[0].structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    subcortex = sum(bm.model[0].index_count for bm in bm if 'SUBCORTICAL' in bm.model[0].structure)
    print(f"Left cortex: {left} vertices")
    print(f"Right cortex: {right} vertices")
    print(f"Subcortex: {subcortex} voxels")
else:
    print("Could not determine brain model structure.")
EOF


# -4-
echo "***** 4: Removing Initial Volumes *****"
# Remove initial transients / artifacts. Equivalent to step 5 in Aradia's pipeline
# Cannot use fslroi because cifti files are not supported by fslroi.
n_remove=5
detrended_trim="${demean_dir}/${fBaseName}_detrend_trim.dtseries.nii"
python3 Dependencies/trim_cifti.py "${detrended}" "${detrended_trim}" ${n_remove}

# -5-
echo "***** 5: Preparing Regressors (Motion, WM, CSF, Global) *****"
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

# -6- (Same as Aradia's step 5b)
echo "***** 6: Demeaning, Detrending, Z-scoring Regressors *****"
regressors_dm="${regressors_dir}/${fBaseName}_regressors_demeaned_detrended.txt"
regressors_z="${regressors_dir}/${fBaseName}_regressors_z.txt"
python3 Aradia/demean_detrend_reg.py -i "${regressors_txt}" -odmdt "${regressors_dm}" -o "${regressors_z}"

# -7- (Same as Aradia's step 5c)
echo "***** 7: Plotting Regressors *****"
reg_png="${plots_dir}/${fBaseName}_regressors_z.png"
python3 Aradia/regressor_plots.py -i "${regressors_z}" -glob yes -o "${reg_png}"

# -8- 
echo "***** 8: Regression, Scrubbing, Bandpass Filtering *****"
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
#!/bin/bash

# Usage: ./pipeline_preprocessing.sh <subject> <session> <task> <direction>
# Example: ./pipeline_preprocessing.sh 01 15 RestingState pa

set -e # Exit on error

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


# -3- Medial wall removal (keep subcortex) using separate/mask/recombine
echo "***** 3: Medial wall removal (keep subcortex) *****"

left_gii="${demean_dir}/${fBaseName}_L.func.gii"
right_gii="${demean_dir}/${fBaseName}_R.func.gii"
subcort_nifti="${demean_dir}/${fBaseName}_subcort.nii.gz"
left_nomw="${demean_dir}/${fBaseName}_L.nomw.func.gii"
right_nomw="${demean_dir}/${fBaseName}_R.nomw.func.gii"

# Separate cifti into left, right, and subcortex
wb_command -cifti-separate "${detrended}" COLUMN \
  -metric CORTEX_LEFT "${left_gii}" \
  -metric CORTEX_RIGHT "${right_gii}" \
  -volume-all "${subcort_nifti}" \
  -label "${demean_dir}/${fBaseName}_subcort_label.nii.gz" \

# Mask cortex with medial wall
wb_command -metric-mask "${left_gii}" "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii" "${left_nomw}"
wb_command -metric-mask "${right_gii}" "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii" "${right_nomw}"

: '
##### Bugfix
wb_command -metric-stats "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii" -column 1 -reduce COUNT_NONZERO
wb_command -metric-stats "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii" -column 1 -reduce COUNT_NONZERO
#python3 <<EOF
#import nibabel as nib
#img = nib.load("${detrended_nomw}")
#print(img.shape)
#EOF

echo -n "Non-zero voxels (subcortical label): "
wb_command -volume-stats "${demean_dir}/${fBaseName}_subcort_label.nii.gz" -reduce COUNT_NONZERO

#echo -n "Non-zero vertices before medial wall mask (left.gii): "
#wb_command -metric-stats "${left_gii}" -column 1 -reduce COUNT_NONZERO
echo -n "Non-zero vertices after medial wall mask (left_nomw): "
wb_command -metric-stats "${left_nomw}" -column 1 -reduce COUNT_NONZERO
python3 <<EOF
import nibabel as nib
m = nib.load("${left_nomw}")
print("Total vertices (left_nomw):", m.darrays[0].data.shape[0])
EOF

#echo -n "Non-zero vertices before medial wall mask (right.gii): "
#wb_command -metric-stats "${right_gii}" -column 1 -reduce COUNT_NONZERO
echo -n "Non-zero vertices after medial wall mask (right_nomw): "
wb_command -metric-stats "${right_nomw}" -column 1 -reduce COUNT_NONZERO
python3 <<EOF
import nibabel as nib
m = nib.load("${right_nomw}")
print("Total vertices (right_nomw):", m.darrays[0].data.shape[0])
EOF

wb_command -metric-stats "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii" -column 1 -reduce COUNT_NONZERO
python3 <<EOF
import nibabel as nib
m = nib.load("${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii")
print("Total vertices (left ROI):", m.darrays[0].data.shape[0])
EOF
wb_command -metric-stats "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii" -column 1 -reduce COUNT_NONZERO
python3 <<EOF
import nibabel as nib
m = nib.load("${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii")
print("Total vertices (right ROI):", m.darrays[0].data.shape[0])
EOF

python3 <<EOF
import nibabel as nib
m = nib.load("${left_nomw}")
print("Total vertices (left_nomw):", m.darrays[0].data.shape[0])
EOF
echo "Left ROI vertex count:"
wb_command -file-information "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii"

python3 <<EOF
import nibabel as nib
m = nib.load("${right_nomw}")
print("Total vertices (right_nomw):", m.darrays[0].data.shape[0])
EOF
echo "Right ROI vertex count:"
wb_command -file-information "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii"
'

# Recombine left, right, and subcortex into a new cifti
detrended_nomw="${demean_dir}/${fBaseName}_detrended_nomw.dtseries.nii"

wb_command -cifti-create-dense-timeseries "${detrended_nomw}" \
  -left-metric "${left_nomw}" \
  -roi-left "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii" \
  -right-metric "${right_nomw}" \
  -roi-right "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii" \
  -volume "${subcort_nifti}" "${demean_dir}/${fBaseName}_subcort_label.nii.gz"

: '
##### Bugfix
# QC: Print file paths and nonzero counts before CIFTI creation
echo "Using files for CIFTI creation:"
echo "Left metric: ${left_nomw}"
echo "Left ROI: ${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii"
echo "Right metric: ${right_nomw}"
echo "Right ROI: ${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii"
echo "Subcortical volume: ${subcort_nifti}"
echo "Subcortical label: ${demean_dir}/${fBaseName}_subcort_label.nii.gz"

echo "Left ROI nonzero vertices:"
wb_command -metric-stats "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii" -column 1 -reduce COUNT_NONZERO
echo "Right ROI nonzero vertices:"
wb_command -metric-stats "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii" -column 1 -reduce COUNT_NONZERO
echo "Subcortical label nonzero voxels:"
wb_command -volume-stats "${demean_dir}/${fBaseName}_subcort_label.nii.gz" -reduce COUNT_NONZERO
'

# Create CIFTI
wb_command -cifti-create-dense-timeseries "${detrended_nomw}" \
  -left-metric "${left_nomw}" \
  -roi-left "${medial_wall_dir}/L.atlasroi.32k_fs_LR.shape.gii" \
  -right-metric "${right_nomw}" \
  -roi-right "${medial_wall_dir}/R.atlasroi.32k_fs_LR.shape.gii" \
  -volume "${subcort_nifti}" "${demean_dir}/${fBaseName}_subcort_label.nii.gz"

# QC: Print output CIFTI shape
python3 <<EOF
import nibabel as nib
img = nib.load("${detrended_nomw}")
print("CIFTI shape:", img.shape)
EOF


# Use the new file for downstream steps
detrended="${detrended_nomw}"
: '
##### Bugfix
echo "***** Output file structure (R, L, subcortex) *****"
echo "Total number of time series (left + right cortex + subcortex):"
python3 <<EOF
import nibabel as nib
img = nib.load("${detrended}")
axis = img.header.get_axis(1)
for tup in axis.iter_structures():
    print(tup)
counts = {}
for name, slc, _ in axis.iter_structures():
    if slc.stop is not None:
        counts[name] = slc.stop - slc.start
    else:
        counts[name] = "unknown"
for k, v in counts.items():
    print(f"{k}: {v}")
print(f"Total: {sum([v for v in counts.values() if isinstance(v, int)])} time series")
EOF
'

# -4-
echo "***** 4: Removing Initial Volumes *****"
# Remove initial transients / artifacts. Equivalent to step 5 in Aradia's pipeline
# Cannot use fslroi because cifti files are not supported by fslroi.
# Something was off here and it reduced the number of voxels instead of timepoints.
# Instead, we will use a custom Python script to trim the CIFTI file.
n_remove=5
detrended_trim="${demean_dir}/${fBaseName}_detrend_trim.dtseries.nii"
python3 /home/hmueller2/ibc_code/ibc_latent/Preprocessing/Dependencies/trim_cifti.py "${detrended}" "${detrended_trim}" ${n_remove}

# -5-
echo "***** 5: Preparing Regressors (Motion, WM, CSF, Global) *****"
# Extract motion parameters
motion_txt="${regressors_dir}/${fBaseName}_motion.txt"
python3 <<EOF
import pandas as pd
conf = pd.read_csv("${confounds}", sep='\t')
motion_cols = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
conf[motion_cols].iloc[${n_remove}:].to_csv("${motion_txt}", sep=' ', index=False, header=False)
EOF

# Extract CSF, WM, and global signal as txt
csf_txt="${regressors_dir}/${fBaseName}_csf.txt"
wm_txt="${regressors_dir}/${fBaseName}_wm.txt"
global_txt="${regressors_dir}/${fBaseName}_global.txt"
python3 <<EOF
import pandas as pd
conf = pd.read_csv("${confounds}", sep='\t')
conf['csf'].iloc[${n_remove}:].to_csv("${csf_txt}", sep=' ', index=False, header=False)
conf['white_matter'].iloc[${n_remove}:].to_csv("${wm_txt}", sep=' ', index=False, header=False)
conf['global_signal'].iloc[${n_remove}:].to_csv("${global_txt}", sep=' ', index=False, header=False)
EOF

# Run friston_regression.py to generate full nuisance regressor file
friston_out="${regressors_dir}/${fBaseName}_friston24.txt"
nuisance_out="${regressors_dir}/${fBaseName}_nuisance_regressors.txt"
python3 /home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia/friston_regression.py \
    -im "${motion_txt}" \
    -i_csf "${csf_txt}" \
    -i_wm "${wm_txt}" \
    -i_glob "${global_txt}" \
    -o "${friston_out}" \
    -o_nuis "${nuisance_out}"


# -6- (Same as Aradia's step 5b)
echo "***** 6: Demeaning, Detrending, Z-scoring Regressors *****"
regressors_dm="${regressors_dir}/${fBaseName}_regressors_demeaned_detrended.txt"
regressors_z="${regressors_dir}/${fBaseName}_regressors_z.txt"
python3 /home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia/demean_detrend_reg.py -i "${nuisance_out}" -odmdt "${regressors_dm}" -o "${regressors_z}"

# -7- (Same as Aradia's step 5c)
echo "***** 7: Plotting Regressors *****"
reg_png="${plots_dir}/${fBaseName}_regressors_z.png"
python3 /home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia/regressor_plots.py -i "${regressors_z}" -glob yes -o "${reg_png}"

# -8- Regression, Scrubbing, Bandpass Filtering
echo "***** 8: Regression, Scrubbing, Bandpass Filtering *****"
TR=$(jq -r '.RepetitionTime' "${json}")
echo "The TR is $TR seconds."

cleaned_bold="${glm_dir}/${fBaseName}_cleaned.dtseries.nii"
python3 -c "import nilearn; print(nilearn.__version__)"
python3 /home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia/regression_interpolation.py \
    -i "${detrended_trim}" \
    -r "${regressors_z}" \
    -FD "${FD_mask}" \
    -TR "${TR}" \
    -o "${cleaned_bold}"

if [ -f "${cleaned_bold}" ]; then
    echo "Cleaned BOLD file created successfully: ${cleaned_bold}"
else
    echo "Error: Cleaned BOLD file was not created."
    exit 1
fi

echo "***** Finalizing Outputs *****"
echo "Pipeline complete. Outputs in ${out_dir}"
#!/bin/bash

studyDataDir=$1
sequenceName=$2
subject=$3
session=$4
run=$5
analysis=$6

fBaseName=${subject}_ses-${session}_${sequenceName}_run-${run}
curDir=$(pwd)/preprocessing

echo "***** 1: Calculating framewise displacement. *****"
# Note: the plotting function goes across runs and even sessions and the scripts are not included here.

FD_threshold=0.2

# mc_dir=${studyDataDir}/derivatives/preprocess_SBREF/${subject}/ses-${session}/motioncorrection_pars # lowres
# FD_input=${mc_dir}/${fBaseName}_bold_mc_reordered.par # lowres

mc_dir=${studyDataDir}/derivatives/preprocess_HR/${subject}/ses-${session}/motioncorrection_pars # highres
FD_input=${mc_dir}/${fBaseName}_mc_reordered.par #highres

output_FD=${mc_dir}/${fBaseName}_fd.txt

python ${curDir}/process_fd.py -i ${FD_input} -FD ${FD_threshold} -o ${output_FD}


echo "***** 2: Demeaning and Detrending EPI. *****"

# input_epi=${studyDataDir}/derivatives/preprocess_SBREF/${subject}/ses-${session}/${fBaseName}_bold_mc.nii # lowres
input_epi=${studyDataDir}/derivatives/preprocess_HR/${subject}/ses-${session}/${fBaseName}_mc.nii # highres

# demean_dir=${studyDataDir}/derivatives/demean_detrend/${subject}/ses-${session} #lowres
demean_dir=${studyDataDir}/derivatives/demean_detrend_HR/${subject}/ses-${session} #highres

demeaned_epi=${demean_dir}/${fBaseName}_demean.nii.gz
meaned_epi=${demean_dir}/${fBaseName}_mean.nii.gz
mkdir -p ${demean_dir}

fslmaths ${input_epi} -Tmean ${meaned_epi} #calculate mean
fslmaths ${input_epi} -sub ${meaned_epi} ${demeaned_epi} #subtract mean from input epi

echo "demeaned std:"
fslstats ${demeaned_epi} -S # quality control: print std


# DETRENDING
detrended_epi=${demean_dir}/${fBaseName}_detrend.nii
3dDetrend -prefix ${fBaseName}_detrend.nii \
        -session ${demean_dir} \
        -verb \
        -polort 1 \
        ${demeaned_epi}

echo "detrended std:"
fslstats ${detrended_epi} -S # quality control: print std


echo "***** 3: Prepare WM, CSF, Global Signal Regression. *****"

# extraction_dir=${studyDataDir}/derivatives/time_course_extraction/${subject}/ses-${session} #lowres
extraction_dir=${studyDataDir}/derivatives/time_course_extraction_HR/${subject}/ses-${session} #highres
mkdir -p ${extraction_dir}

# ADD DIRECTORY TO GDC HERE TO APPLY WARPS TO THE MASKS
# gdc_dir=${studyDataDir}/derivatives/ref_anat_fullbrain_gdc_topup/${subject}/ses-${session}/run-${run} #lowres
gdc_dir=${studyDataDir}/derivatives/${analysis}/${subject}/ses-${session} #highres
aseg_dir=${studyDataDir}/derivatives/mprage_recon-all_gdc/${subject}/freesurfer/mri 


# CONVERT ASEG TO NII
# These steps caused substantial interferences because many of the parallel processes will try to work on the same files, bc they are structural
# trying flock
# not tested yet!!

# echo "trying lock"
# LOCKFILE="/tmp/my_script_${subject}_${session}.lock" # want to lock each session bc gdc is session specific, but not run-specific


# (
#   flock -n 200 || { echo "Another instance is running. Waiting..."; flock 200; }

#   # Critical section: create file only if it does not exist
#   if [ ! -e "${gdc_dir}/aseg.nii.gz" ]; then
#       echo "File does not exist. Creating file..."
      
#       mri_convert ${aseg_dir}/aseg.mgz ${gdc_dir}/aseg.nii.gz
      
#       echo "3a. Deriving WM Masks."
#       # WM is 2 (L) and 41 (R) in freesurfer
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 2 -uthr 2 -bin ${gdc_dir}/left_wm_mask.nii.gz
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 41 -uthr 41 -bin ${gdc_dir}/right_wm_mask.nii.gz
#       # add left and right together
#       fslmaths ${gdc_dir}/left_wm_mask.nii.gz -add ${gdc_dir}/right_wm_mask.nii.gz ${gdc_dir}/combined_wm_mask.nii.gz

#       echo "3b. Deriving CSF Masks."
#       # CSF:
#       # 4 Left Lateral Ventricle
#       # 5 Left-Inf-Lat-Vent
#       # 14 3rd Ventricle
#       # 15 4th Ventricle
#       # 24 CSF
#       # 43 Right Lateral Ventricle
#       # 44 Right-Inf-Lat-Vent
#       # 30 Left-Vessel
#       # 31 Left choroid plexus
#       # 62 Right Vessel
#       # 63 Right choroid plexus
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 4 -uthr 5 -bin ${gdc_dir}/left_lat_vent_mask.nii.gz
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 14 -uthr 15 -bin ${gdc_dir}/third_fourth_vent_mask.nii.gz
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 24 -uthr 24 -bin ${gdc_dir}/csf_part_mask.nii.gz
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 43 -uthr 44 -bin ${gdc_dir}/right_lat_vent_mask.nii.gz
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 30 -uthr 31 -bin ${gdc_dir}/left_vessel_cp_mask.nii.gz
#       fslmaths ${gdc_dir}/aseg.nii.gz -thr 62 -uthr 63 -bin ${gdc_dir}/right_vessel_cp_mask.nii.gz

#       echo "Merging CSF Masks."
#       # merge them all together to one mask
#       fslmaths ${gdc_dir}/left_lat_vent_mask.nii.gz -add ${gdc_dir}/third_fourth_vent_mask.nii.gz -add ${gdc_dir}/csf_part_mask.nii.gz -add ${gdc_dir}/right_lat_vent_mask.nii.gz -add ${gdc_dir}/left_vessel_cp_mask.nii.gz -add ${gdc_dir}/right_vessel_cp_mask.nii.gz ${gdc_dir}/csf_mask.nii.gz
#       echo "3c. Warping masks."

#       # put masks in func space -> mask, csf, wm
#       # put in ref_anat folder

#       # Lowres 
# #       echo "CSF mask."
# #       applywarp -i ${gdc_dir}/csf_mask.nii.gz --interp=nn -o ${gdc_dir}/csf_mask_in_func_pre -r ${gdc_dir}/funcmean -w ${gdc_dir}/fs_T1_2_funcmean_warpfield --abs

# #       echo "WM mask."
# #       applywarp -i ${gdc_dir}/combined_wm_mask.nii.gz --interp=nn -o ${gdc_dir}/wm_mask_in_func_pre -r ${gdc_dir}/funcmean -w ${gdc_dir}/fs_T1_2_funcmean_warpfield --abs
      
# #       echo "Brain mask."
# #       mri_convert ${aseg_dir}/brainmask_mask.mgz ${gdc_dir}/brainmask_mask.nii.gz
# #       applywarp -i ${gdc_dir}/brainmask_mask.nii.gz --interp=nn -o ${gdc_dir}/brain_mask_in_func_pre -r ${gdc_dir}/funcmean -w ${gdc_dir}/fs_T1_2_funcmean_warpfield --abs


#       # Highres
#       echo "CSF mask."
#       applywarp -i ${gdc_dir}/csf_mask.nii.gz --interp=nn -o ${gdc_dir}/csf_mask_in_func_pre -r ${gdc_dir}/slab -w ${gdc_dir}/fs_T1_2_slab_warpfield --abs

#       echo "WM mask."
#       applywarp -i ${gdc_dir}/combined_wm_mask.nii.gz --interp=nn -o ${gdc_dir}/wm_mask_in_func_pre -r ${gdc_dir}/slab -w ${gdc_dir}/fs_T1_2_slab_warpfield --abs
      
#       echo "Brain mask."
#       mri_convert ${aseg_dir}/brainmask_mask.mgz ${gdc_dir}/brainmask_mask.nii.gz
#       applywarp -i ${gdc_dir}/brainmask_mask.nii.gz --interp=nn -o ${gdc_dir}/brain_mask_in_func_pre -r ${gdc_dir}/slab -w ${gdc_dir}/fs_T1_2_slab_warpfield --abs


#   else
#       echo "File already exists. Skipping creation."
#   fi

# ) 200>"$LOCKFILE"



# calculate mean WM, CSF signal in epi timeseries for phys regression
# use fslmeants

echo "Calculating mean of WM, CSF."

# White Matter
fslmeants -i ${detrended_epi} -v -o ${extraction_dir}/${fBaseName}_mean_wm.txt -m ${gdc_dir}/wm_mask_in_func_pre.nii

# CSF
fslmeants -i ${detrended_epi} -v -o ${extraction_dir}/${fBaseName}_mean_csf.txt -m ${gdc_dir}/csf_mask_in_func_pre.nii


# echo "3d. Global signal extraction."

# calculate mean of all voxels
# probs put a different input file here depending on the pipeline

fslmeants -i ${detrended_epi} -v -o ${extraction_dir}/${fBaseName}_global_mean.txt -m ${gdc_dir}/brain_mask_in_func_pre.nii


# File cleaning
# echo "3e. Finished processing, now cleaning."
# rm ${gdc_dir}/left_wm_mask.nii.gz
# rm ${gdc_dir}/right_wm_mask.nii.gz
# rm ${gdc_dir}/aseg.nii.gz
# rm ${gdc_dir}/left_lat_vent_mask.nii.gz
# rm ${gdc_dir}/third_fourth_vent_mask.nii.gz
# rm ${gdc_dir}/csf_part_mask.nii.gz
# rm ${gdc_dir}/right_lat_vent_mask.nii.gz
# rm ${gdc_dir}/left_vessel_cp_mask.nii.gz
# rm ${gdc_dir}/right_vessel_cp_mask.nii.gz



echo "***** 4: Calculating regressors: Volterra expansion. *****"

wm_mean=${extraction_dir}/${fBaseName}_mean_wm.txt
csf_mean=${extraction_dir}/${fBaseName}_mean_csf.txt
global_mean=${extraction_dir}/${fBaseName}_global_mean.txt

# regressors_dir=${studyDataDir}/derivatives/regressors/${subject}/ses-${session} #lowres
regressors_dir=${studyDataDir}/derivatives/regressors_HR/${subject}/ses-${session} #highres
mkdir -p ${regressors_dir}

friston_parameters=${regressors_dir}/${fBaseName}_friston_motion_regressors.txt
all_regressors=${regressors_dir}/${fBaseName}_all_regressors.txt

# this will calculate the volterra expansion and combine all regressors to one file
python ${curDir}/friston_regression.py -im ${FD_input} -i_csf ${csf_mean} -i_wm ${wm_mean} -i_glob None -o ${friston_parameters} -o_nuis ${all_regressors} # MODIFIED THIS FUNCTION FOR HIGHRES


echo "***** 5: Calculating GLM. *****"

echo "5a. Removing first 5 volumes."
# i: regressors
clean_regs=${regressors_dir}/run-${run}_all_regressors_cleaned.txt
python ${curDir}/remove_volumes.py -i ${all_regressors} -v 5 -o ${clean_regs}
# ii: epi tseries
epi_clean=${demean_dir}/${fBaseName}_detrend_cleaned.nii

# not all participants have the exact same number of volumes, so extract the number of volumes directly from the nifti file to make more robust
# also, high res has 300, while low res has 600 
n_vols=$(fslhd ${detrended_epi} | grep '^dim4' | awk '{print $2}')
echo ${n_vols}
new_vols=$((n_vols - 5))
echo ${new_vols}

# remove first 5 volumes
fslroi ${detrended_epi} ${epi_clean} 5 ${new_vols}

echo "5b. Demeaning, Detrending, z-scoring."
dm_reg=${regressors_dir}/${fBaseName}_all_regressors_demeaned_detrended.txt # this step is not really necessary to save as a separate file, but for completeness
z_reg=${regressors_dir}/${fBaseName}_all_regressors_demeaned_detrended_z-scored.txt
python ${curDir}/demean_detrend_reg.py -i ${clean_regs} -odmdt ${dm_reg} -o ${z_reg}

echo "5c. Plotting the regressors."
reg_png=${regressors_dir}/${fBaseName}_all_regressors_demeaned_detrended_z-scored.png
# python ${curDir}/regressor_plots.py -i ${z_reg} -glob yes -o ${reg_png} # lowres
python ${curDir}/regressor_plots.py -i ${z_reg} -glob no -o ${reg_png} # high res

echo "5d. Regression, Scrubbing, Interpolating, Bandpass Filtering."

# extract TR from the json file
json=${studyDataDir}/${subject}/ses-${session}/func/${fBaseName}_bold.json # lowres and highres same
TR=$(jq -r '.RepetitionTime' ${json})
echo "The TR is $TR seconds."

# regression_dir=${studyDataDir}/derivatives/regression_nilearn/${subject}/ses-${session} # lowres
regression_dir=${studyDataDir}/derivatives/regression_nilearn_HR/${subject}/ses-${session}
mkdir -p ${regression_dir}
processed_epi=${regression_dir}/${fBaseName}_cleaned.nii

# clean the FD of the first 5 volumes before scrubbing
clean_FD=${mc_dir}/${fBaseName}_fd_without_first_5.txt
python ${curDir}/remove_volumes.py -i ${output_FD} -v 5 -o ${clean_FD}

echo "i. Running cleaning."
python ${curDir}/regression_interpolation.py -i ${epi_clean}.gz -r ${z_reg} -FD ${clean_FD} -TR ${TR} -o ${processed_epi}

# echo "ii. Adding mean."
# fslmaths ${processed_epi} -add ${meaned_epi} ${regression_dir}/${fBaseName}_cleaned_plus_mean.nii

echo "post-regression std:"
fslstats ${processed_epi} -S


# 5e: maybe: visualize GLM results?
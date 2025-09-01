% filepath: /home/hmueller2/ibc_code/ibc_latent/Infomap/Utilities/test_make_dmat.m
% Minimal test for pfm_make_dmat_new

% Add dependencies to Matlab search path & github repo of MSC
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/Infomap/Utilities'));
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/MSCcodebase/Utilities'))

% Set paths to your files
cifti_file = '/ptmp/hmueller2/Downloads/individual_networks/sub-01/resting_state/sub-01_ses-15_resting_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii';
lh_surf_file = '/ptmp/hmueller2/Downloads/fmriprep_out/sub-01/anat/sub-01_hemi-L_midthickness.32k_fs_LR.surf.gii';
rh_surf_file = '/ptmp/hmueller2/Downloads/fmriprep_out/sub-01/anat/sub-01_hemi-R_midthickness.32k_fs_LR.surf.gii';
out_dir = '/ptmp/hmueller2/Downloads/individual_networks/sub-01/resting_state/';
nWorkers = 16; % or whatever you want
WorkbenchBinary = '/mnt/workbench/run_wb_command.sh';

% Prepare input cell array for surfaces
MidthickSurfs = {lh_surf_file, rh_surf_file};
disp('All files set.');

% Run only the distance matrix step
pfm_make_dmat_new(cifti_file, MidthickSurfs, out_dir, nWorkers, WorkbenchBinary);

% After running, check the output
dm = load([out_dir '/DistanceMatrix.mat']);
disp(['Distance matrix shape: ' mat2str(size(dm.D))]);
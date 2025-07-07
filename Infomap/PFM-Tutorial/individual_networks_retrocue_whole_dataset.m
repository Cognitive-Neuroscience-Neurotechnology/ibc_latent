%% A tutorial covering precision functional mapping using an example dataset. 

%% Before you begin.

% add dependencies to Matlab search path
addpath(genpath('/home/mawilms/retrocue_resting/analysis/individual_networks/PFM-Depression/PFM-Tutorial/Utilities'));

% add github repo of MC
addpath(genpath('/home/mawilms/retrocue_resting/analysis/individual_networks/MSCcodebase/Utilities'))

% define path to some software packages that will be needed
InfoMapBinary = '/home/mawilms/retrocue_resting/packages/Infomap'; % path to infomap binary; code tested on version 2.0.0 
WorkbenchBinary = '/home/mawilms/retrocue_resting/packages/workbench/bin_linux64/wb_command'; % path to workbench binary; code tested on version 1.4.2

% number of 
% workers
nWorkers = 30;

%% Step 1: Temporal Concatenation of fMRI data from all sessions.

% define subject directory and name;
derivative_dir = '/ptmp/mawilms/retrocue_resting_Nyx/derivatives';
Subject=getenv('Subject');

% define & create
% the pfm directory;
Subdir = [derivative_dir '/individual_networks/sub-' Subject];
mkdir(Subdir);

tseries_dir=[derivative_dir '/extracted_tseries_parc/sub-' Subject];
ciftify_dir=[derivative_dir '/ciftify_32k/sub-' Subject];

% define fs_lr_32k midthickness surfaces;
MidthickSurfs{1} = [ciftify_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.L.midthickness.32k_fs_LR.surf.gii'];
MidthickSurfs{2} = [ciftify_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.R.midthickness.32k_fs_LR.surf.gii'];

left_mask=[ciftify_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.L.atlasroi.32k_fs_LR.shape.gii'];
right_mask=[ciftify_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.R.atlasroi.32k_fs_LR.shape.gii'];



half_dir=[derivative_dir '/individual_networks/sub-' Subject '/whole_dataset'];
mkdir(half_dir);

% preallocate;
ConcatenatedData = [];

% count the number of imaging sessions;
dirInfo = dir([tseries_dir '/ses-*']); % get all directories
directories = {dirInfo([dirInfo.isdir]).name}; 
disp(directories)

nSessions = length(directories); 
disp(nSessions)

% sweep through
% the sessions;
for i = 1:nSessions
    current_ses=directories{i};
    disp(current_ses)

    % count the number of runs in this session
    nRuns = length(dir([tseries_dir '/' current_ses '/*run-*.32k.dtseries.nii']));
    disp(nRuns)
    % sweep 
    % through
    % the runs;
    for ii = 1:nRuns
        % load the CIFTI file for run "ii" 
        current_file=[tseries_dir '/' current_ses '/sub-' Subject '_' current_ses '_task-rest_acq-lowresmb_run-' sprintf('%d',ii) '.LR.32k.dtseries.nii'];
        disp(current_file)
        if exist(current_file, 'file') ~= 2
            error(['File does not exist: ' current_file]);
        end

        Cifti = ft_read_cifti_mod(current_file);
        ConcatenatedData = [ConcatenatedData Cifti.data(:,:)];

    end

end

% make a single CIFTI containing 
% time-series from all scans;
ConcatenatedCifti = Cifti;
ConcatenatedCifti.data = ConcatenatedData;

% Step 2: Make a distance matrix.


% make the distance matrix;
disp('Making dmat')
tic;
pfm_make_dmat(ConcatenatedCifti,MidthickSurfs,half_dir,nWorkers,WorkbenchBinary);
elapsedTime = toc;
disp(['Elapsed time: ', num2str(elapsedTime), ' seconds'])
disp('Done with dmat')


% write out the CIFTI file;
concat_file=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_LR_32k.dtseries.nii'];
disp(concat_file)
ft_write_cifti_mod(concat_file, ConcatenatedCifti);

% Step 3: Apply spatial smoothing.

% define a range of gaussian 
% smoothing kernels (in sigma)
KernelSizes = [0.85]; %2mm

% intermediate files:
left_metric=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_LR_32k.left.func.gii'];
right_metric=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_LR_32k.right.func.gii'];

% prepare files for smoothing
% generate func.gii from dtseries to use a different smoothing method than Lynch et al. 2024 
system([WorkbenchBinary ' -cifti-separate ' concat_file ' COLUMN -metric CORTEX_LEFT ' left_metric ' -metric CORTEX_RIGHT ' right_metric]);


% sweep a range of
% smoothing kernels;
for k = KernelSizes

    % smooth with geodesic (for surface data) and Euclidean (for volumetric data) Gaussian kernels;
    smoothed_left=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_LR_32k.left.' num2str(k) '.func.gii'];
    smoothed_right=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_LR_32k.right.' num2str(k) '.func.gii'];
    system([WorkbenchBinary ' -metric-smoothing ' MidthickSurfs{1} ' ' left_metric ' 0.85 ' smoothed_left]);
    system([WorkbenchBinary ' -metric-smoothing ' MidthickSurfs{2} ' ' right_metric ' 0.85 ' smoothed_right]);

    % go back to cifti dtseries
    smoothed_file=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_smoothed' num2str(k) '_32k_fsLR.dtseries.nii'];
    system([WorkbenchBinary ' -cifti-create-dense-timeseries ' smoothed_file ' -left-metric ' smoothed_left ' -right-metric ' smoothed_right]);

    % remove medial wall
    smoothed_masked_file=[half_dir '/sub-' Subject '_task-rest_acq-lowresmb_concatenated_smoothed' num2str(k) '_masked_32k_fsLR.dtseries.nii'];
    system([WorkbenchBinary ' -cifti-restrict-dense-map ' smoothed_file ' COLUMN ' smoothed_masked_file ' -left-roi ' left_mask ' -right-roi ' right_mask]);


end
disp('Finished smoothing.')


% Step 4: Run infomap.

% load your concatenated resting-state dataset, pick whatever level of spatial smoothing you want
ConcatenatedCifti = ft_read_cifti_mod(smoothed_masked_file);

% define inputs;
DistanceMatrix = [half_dir '/DistanceMatrix.mat']; % can be path to file
DistanceCutoff = 10; % in mm; usually between 10 to 30 mm works well.
GraphDensities = flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]); % 
NumberReps = 50; % number of times infomap is run;
BadVertices = []; % optional, but you could include regions to ignore, if you know there is bad signal there.
Structures = {'CORTEX_LEFT','CORTEX_RIGHT'};

% run infomap
disp('starting infomap.')
tic;
pfm_infomap(ConcatenatedCifti,DistanceMatrix,half_dir,GraphDensities,NumberReps,DistanceCutoff,BadVertices,Structures,nWorkers,InfoMapBinary);
elapsedTime = toc;
disp(['Elapsed time: ', num2str(elapsedTime), ' seconds'])

% remove some intermediate files (optional)
system(['rm ' half_dir '/*.net']);
system(['rm ' half_dir '/*.clu']);
system(['rm ' half_dir '/*Log*']);

% define inputs;
Input = [half_dir '/Bipartite_PhysicalCommunities.dtseries.nii'];
Output = 'Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii';
MinSize = 50; % in mm^2

% perform spatial filtering
disp('spatial filtering.')
pfm_spatial_filtering(Input,half_dir,Output,MidthickSurfs,MinSize,WorkbenchBinary);

% Step 5: Algorithmic assignment of network identities to infomap communities.

% load the priors;
load('priors.mat'); % FOR THIS WE WILL LATER NEED TO ADAPT WITH THE NETWORKS WE WANT

% define inputs;
Ic = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii']);
Output = 'Bipartite_PhysicalCommunities+AlgorithmicLabeling';
Column = 6; % column 6, representing graph density 0.01% in this example.     

% run the network identification algorithm;
disp('identifying networks.')
pfm_identify_networks(ConcatenatedCifti,Ic,MidthickSurfs,Column,Priors,Output,half_dir,WorkbenchBinary);


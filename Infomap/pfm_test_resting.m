%% A tutorial covering precision functional mapping using an example dataset.
%% This was written by Charles Lynch, PhD, in 2023.
%% This code is part of the PFM-Tutorial repository, available at https://github.com/cjl2007/PFM-Depression

%% --- Before you begin.

% add dependencies to Matlab search path
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/Infomap/Utilities'));

% add github repo of MSC
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/MSCcodebase/Utilities'))

% define path to some software packages that will be needed
InfoMapBinary = '/home/hmueller2/.local/bin/infomap'; % path to infomap binary; code tested on version 2.0.0 
WorkbenchBinary = '/home/hmueller2/workbench/bin_linux64/wb_command'; % path to workbench binary; code tested on version 1.4.2

% number of 
% workers
nWorkers = 5;

%% Step 1: Temporal Concatenation of fMRI data from all sessions.

% define subject directory and name;
Subject = '01';
Session = '15';
working_dir = '/ptmp/hmueller2/Downloads';

tseries_dir = [working_dir '/fmriprep_out/sub-' Subject '/ses-' Session '/postfmriprep/GLM']; 

% Only use the two specified resting-state files
files = {
    [tseries_dir '/sub-01_ses-15_task-RestingState_dir-ap_cleaned.dtseries.nii']
    [tseries_dir '/sub-01_ses-15_task-RestingState_dir-pa_cleaned.dtseries.nii']
};

% Output directories
Subdir = [working_dir '/individual_networks/sub-' Subject];
half_dir = [Subdir '/resting_state'];
mkdir(Subdir); mkdir(half_dir);

% ---- TO FIND!!

% define fs_lr_32k midthickness surfaces;
surface_dir=[working_dir '/fmriprep_out/sub-' Subject]; 

% Loads MidthickSurfs (left and right) to compute geodesic distances on the cortical mesh
MidthickSurfs{1} = [surface_dir '/anat/sub-01_hemi-L_midthickness.surf.gii']; % Found in fmriprep output
MidthickSurfs{2} = [surface_dir '/anat/sub-01_hemi-R_midthickness.surf.gii']; % Found in fmriprep output

% left_mask=[surface_dir '/anat/sub-01_hemi-L_atlasroi.shape.gii']; % Searching
% right_mask=[surface_dir '/anat/sub-01_hemi-R_atlasroi.shape.gii']; % Searching

disp('Some message'); drawnow;

% ---- TO FIND!!

try
    % Concatenate the two files
    ConcatenatedData = [];
    for i = 1:length(files)
        disp(['Loading: ' files{i}])
        Cifti = ft_read_cifti_mod(files{i});
        ConcatenatedData = [ConcatenatedData Cifti.data(:,:)];
    end
    % make a single CIFTI containing time-series from all scans;
    ConcatenatedCifti = Cifti;
    ConcatenatedCifti.data = ConcatenatedData;
    disp(size(ConcatenatedCifti.data));
    disp(MidthickSurfs);
    disp(['Number of grayordinates: ' num2str(size(ConcatenatedCifti.data,1))]);
catch ME
    disp('Error during concatenation of CIFTI files:');
    disp(ME.message);
    return
end

try
    % Step 2: Make a distance matrix.
    disp('Making dmat')
    tic;
    pfm_make_dmat(ConcatenatedCifti,MidthickSurfs,half_dir,nWorkers,WorkbenchBinary);
    disp(['Elapsed time: ', num2str(toc), ' seconds'])
catch ME
    disp('Error during distance matrix creation:');
    disp(ME.message);
    return
end

try
    % optional: regress adjacent cortical signal from subcortex
    [ConcatenatedCifti] = pfm_xregress_adjacent_cortex(ConcatenatedCifti,[half_dir '/DistanceMatrix.mat'],20);
    % write out the CIFTI file;
    concat_file=[half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_fsLR.dtseries.nii'];
    disp(concat_file)
    ft_write_cifti_mod(concat_file, ConcatenatedCifti);
catch ME
    disp('Error during regression or writing CIFTI:');
    disp(ME.message);
    return
end

%% Step 3: Apply spatial smoothing.

% define a range of gaussian smoothing kernels (in sigma)
k = 0.85;
smoothed_file = [half_dir '/sub-01_ses-15_resting_concatenated_cleaned_smoothed_' num2str(k) '_fsLR.dtseries.nii'];
system([WorkbenchBinary ' -cifti-smoothing ' concat_file ' ' num2str(k) ' ' num2str(k) ' COLUMN ' smoothed_file ...
    ' -left-surface ' MidthickSurfs{1} ' -right-surface ' MidthickSurfs{2} ' -merged-volume']);

%% Step 4: Run infomap.
% load your concatenated resting-state dataset, pick whatever level of spatial smoothing you want
ConcatenatedCifti = ft_read_cifti_mod(smoothed_file);

% define inputs;
DistanceMatrix = [half_dir '/DistanceMatrix.mat']; % can be path to file
DistanceCutoff = 10; % in mm; usually between 10 to 30 mm works well.
GraphDensities = flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]); % 
NumberReps = 5; % Fewer reps for speed
BadVertices = []; % optional, but you could include regions to ignore, if you know there is bad signal there.
Structures = {'CORTEX_LEFT','CEREBELLUM_LEFT','ACCUMBENS_LEFT','CAUDATE_LEFT','PALLIDUM_LEFT','PUTAMEN_LEFT','THALAMUS_LEFT','HIPPOCAMPUS_LEFT','AMYGDALA_LEFT','ACCUMBENS_LEFT','CORTEX_RIGHT','CEREBELLUM_RIGHT','ACCUMBENS_RIGHT','CAUDATE_RIGHT','PALLIDUM_RIGHT','PUTAMEN_RIGHT','THALAMUS_RIGHT','HIPPOCAMPUS_RIGHT','AMYGDALA_RIGHT','ACCUMBENS_RIGHT'};

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

disp(['LH vertices: ' num2str(length(LH.vertices))]);
disp(['RH vertices: ' num2str(length(RH.vertices))]);
disp(['Total surface vertices: ' num2str(length(LH.vertices) + length(RH.vertices))]);
disp(['Length of RefCifti.brainstructure: ' num2str(length(RefCifti.brainstructure))]);

%% Step 5: Algorithmic assignment of network identities to infomap communities.

% load the priors;
load('priors.mat'); % FOR THIS WE WILL LATER NEED TO ADAPT WITH THE NETWORKS WE WANT

% define inputs;
Ic = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii']);
Output = 'Bipartite_PhysicalCommunities+AlgorithmicLabeling';
Column = 6; % column 6, representing graph density 0.01% in this example.

% run the network identification algorithm;
disp('identifying networks.')
pfm_identify_networks(ConcatenatedCifti,Ic,MidthickSurfs,Column,Priors,Output,half_dir,WorkbenchBinary);


%{
%% Step 6: Review algorithmic network assignments, optionally adjust labels manually if needed.

% define inputs
XLS = [half_dir '/Bipartite_PhysicalCommunities+AlgorithmicLabeling_NetworkLabels+ManualDecisions.xls']; 
Output = 'Bipartite_PhysicalCommunities+FinalLabeling';

% OPTIONAL: update network assignments according to manual decisioans;
pfm_parse_manual_decisions(Ic,Column,MidthickSurfs,Priors,XLS,Output,PfmDir,WorkbenchBinary);



%% Step 7: Calculate size of each functional brain network

% define inputs
FunctionalNetworks = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+FinalLabeling.dlabel.nii']);
VA = ft_read_cifti_mod([Subdir '/fs_LR/fsaverage_LR32k/' Subject '.midthickness_va.32k_fs_LR.dscalar.nii']);
Structures = {'CORTEX_LEFT','CORTEX_RIGHT'}; % in this case, cortex only.

% calculate the size of each functional brain network
NetworkSize = pfm_calculate_network_size(FunctionalNetworks,VA,Structures);

close all; % blank slate
H = figure; % prellocate parent figure
set(H,'position',[1 1 325 400]); hold;

% unique functional networks;
uCi = unique(nonzeros(FunctionalNetworks.data));

% sweep through
% the networks;
for i = 1:length(uCi)
    Tmp = nan(1,length(Priors.NetworkLabels));
    Tmp(i) = NetworkSize(i);
    barh(Tmp,'FaceColor',Priors.NetworkColors(i,:));
    text((NetworkSize(i)+0.1),i,[num2str(NetworkSize(i),3) '%']);
end

% make it pretty;
yticklabels(Priors.NetworkLabels); 
yticks(1:length(uCi)); ylim([0 21]);
xlim([0 20]); xticks(0:5:20);
set(gca,'fontname','arial','fontsize',10,'TickLength',[0 0],'TickLabelInterpreter','none');
xlabel('% of Cortical Surface');
print(gcf,[PfmDir '/FunctionalNetworkSizes'],'-dpdf');
%}

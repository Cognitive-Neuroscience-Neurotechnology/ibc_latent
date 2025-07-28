%% A tutorial covering precision functional mapping using an example dataset.
%% This was written by Charles Lynch, PhD, in 2023.
%% This code is part of the PFM-Tutorial repository, available at https://github.com/cjl2007/PFM-Depression

%% --- Before you begin.

% add dependencies to Matlab search path
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/PFM-Infomap/Utilities'));

% define path to some software packages that will be needed
InfoMapBinary = '/home/hmueller2/.local/bin/infomap'; % path to infomap binary; code tested on version 2.0.0 
WorkbenchBinary = '/home/hmueller2/workbench/bin_linux64/wb_command'; % path to workbench binary; code tested on version 1.4.2

% number of 
% workers
nWorkers = 30;

%% Step 1: Temporal Concatenation of fMRI data from all sessions.

% define subject directory and name;
working_dir = '/ptmp/hmueller2/Downloads';
Subject=getenv('Subject');

tseries_dir=[working_dir '/fmriprep_out/sub-' Subject]; 

% define and create the pfm directory;
Subdir = [working_dir '/individual_networks/sub-' Subject]; 
mkdir(Subdir);


% ---- TO FIND!!

% define fs_lr_32k midthickness surfaces;
surface_dir=[working_dir '/fmriprep_out/sub-' Subject]; 

MidthickSurfs{1} = [surface_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.L.midthickness.32k_fs_LR.surf.gii'];
MidthickSurfs{2} = [surface_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.R.midthickness.32k_fs_LR.surf.gii'];

left_mask=[surface_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.L.atlasroi.32k_fs_LR.shape.gii'];
right_mask=[surface_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.R.atlasroi.32k_fs_LR.shape.gii'];

% ---- TO FIND!!


half_dir=[working_dir '/individual_networks/sub-' Subject '/whole_dataset'];
mkdir(half_dir);

% count the number of imaging sessions;
dirInfo = dir([tseries_dir '/ses-*']); % get all directories
directories = {dirInfo([dirInfo.isdir]).name}; 
disp(directories)

nSessions = length(directories); 
disp(nSessions)

% sweep through the sessions;
ConcatenatedData = [];
for i = 1:nSessions
    current_ses = directories{i};
    disp(current_ses)
    GLMdir = [tseries_dir '/' current_ses '/postfmriprep/GLM/'];
    % Find all cleaned CIFTI files in this session
    files = dir([GLMdir 'sub-' Subject '_' current_ses '_task-*_dir-*_cleaned.dtseries.nii']);
    disp(['Found ' num2str(length(files)) ' runs in ' current_ses]);
    for f = 1:length(files)
        current_file = fullfile(GLMdir, files(f).name);
        disp(current_file)
        Cifti = ft_read_cifti_mod(current_file);
        ConcatenatedData = [ConcatenatedData Cifti.data(:,:)];
    end
end

% make a single CIFTI containing time-series from all scans;
ConcatenatedCifti = Cifti;
ConcatenatedCifti.data = ConcatenatedData;



%% Step 2: Make a distance matrix.

% make the distance matrix;
disp('Making dmat')
tic;
pfm_make_dmat(ConcatenatedCifti,MidthickSurfs,half_dir,nWorkers,WorkbenchBinary);
elapsedTime = toc;
disp(['Elapsed time: ', num2str(elapsedTime), ' seconds'])
disp('Done with dmat')

% optional: regress adjacent cortical signal from subcortex to reduce artifactual coupling 
% (for example, between cerebellum and visual cortex, or between putamen and insular cortex)
[ConcatenatedCifti] = pfm_regress_adjacent_cortex(ConcatenatedCifti,[half_dir '/DistanceMatrix.mat'],20);

% write out the CIFTI file;
concat_file=[half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_fsLR.dtseries.nii'];
disp(concat_file)
ft_write_cifti_mod(concat_file, ConcatenatedCifti);



%% Step 3: Apply spatial smoothing.

% define a range of gaussian smoothing kernels (in sigma)
KernelSizes = [0.85 1.7 2.55];

% sweep a range of smoothing kernels;
for k = KernelSizes
    % smooth with geodesic (for surface data) and Euclidean (for volumetric data) Gaussian kernels;
    smoothed_file = [half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_smoothed_' num2str(k) '_fsLR.dtseries.nii'];
    system([WorkbenchBinary ' -cifti-smoothing ' ...
        half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_fsLR.dtseries.nii ' ...
        num2str(k) ' ' num2str(k) ' COLUMN ' smoothed_file ...
        ' -left-surface ' MidthickSurfs{1} ' -right-surface ' MidthickSurfs{2} ' -merged-volume']);
end



%% Step 4: Run infomap.

% load your concatenated resting-state dataset, pick whatever level of spatial smoothing you want
ConcatenatedCifti = ft_read_cifti_mod([half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii']);

% define inputs;
DistanceMatrix = [half_dir '/DistanceMatrix.mat']; % can be path to file
DistanceCutoff = 10; % in mm; usually between 10 to 30 mm works well.
GraphDensities = flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]); % 
NumberReps = 50; % number of times infomap is run;
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

%% A tutorial covering precision functional mapping using an example dataset.
%% This was written by Charles Lynch, PhD, in 2023.
%% This code is part of the PFM-Tutorial repository, available at https://github.com/cjl2007/PFM-Depression

%% ---- Before you begin ---- 

% Add dependencies to Matlab search path & github repo of MSC
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/Infomap/Utilities'));
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/MSCcodebase/Utilities'))

InfoMapBinary = '/usr/local/bin/infomap';
WorkbenchBinary = '/mnt/workbench/run_wb_command.sh'; % Used in order to use wb_command in apptainer container
nWorkers = 16;

% Get subject from SLURM script/environment. E.g. Subject='01'

working_dir = '/ptmp/hmueller2/Downloads';


%% ---- Step 1: Temporal Concatenation of fMRI data from all sessions.
% left_mask=[surface_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.L.atlasroi.32k_fs_LR.shape.gii'];
% right_mask=[surface_dir '/freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.R.atlasroi.32k_fs_LR.shape.gii'];

disp(['Processing subject: ' Subject]);
tseries_dir = [working_dir '/fmriprep_out'];
surface_dir = [tseries_dir '/sub-' Subject];

% Find all session directories for this subject
ses_dirs = dir([tseries_dir '/sub-' Subject '/ses-*']);
session_names = {ses_dirs([ses_dirs.isdir]).name};

ConcatenatedData = [];
for i = 1:length(session_names)
    Session = session_names{i};
    GLMdir = [tseries_dir '/sub-' Subject '/' Session '/postfmriprep/GLM/'];
    files = dir([GLMdir 'sub-' Subject '_' Session '_task-*_dir-*_cleaned.dtseries.nii']);
    for f = 1:length(files)
        current_file = fullfile(GLMdir, files(f).name);
        disp(['Loading: ' current_file])
        Cifti = ft_read_cifti_mod(current_file);
        ConcatenatedData = [ConcatenatedData Cifti.data(:,:)];
    end
end

% Use the last loaded Cifti as template
ConcatenatedCifti = Cifti;
ConcatenatedCifti.data = ConcatenatedData;

% Output directories
Subdir = [working_dir '/individual_networks/sub-' Subject];
half_dir = [Subdir '/whole_dataset'];
mkdir(Subdir); mkdir(half_dir);

% Output directories
Subdir = [working_dir '/individual_networks/sub-' Subject];
half_dir = [Subdir '/whole_dataset'];
mkdir(Subdir); mkdir(half_dir);

% Surface files
MidthickSurfs{1} = [surface_dir '/anat/sub-' Subject '_hemi-L_midthickness.32k_fs_LR.surf.gii'];
MidthickSurfs{2} = [surface_dir '/anat/sub-' Subject '_hemi-R_midthickness.32k_fs_LR.surf.gii'];


%% ---- Step 2: Make a distance matrix.
disp('Making dmat')
tic;
pfm_make_dmat(ConcatenatedCifti,MidthickSurfs,half_dir,nWorkers,WorkbenchBinary);
elapsed_minutes = toc / 60;
disp(['Elapsed time: ', num2str(elapsed_minutes, '%.2f'), ' minutes'])

% Optional: regress adjacent cortical signal from subcortex to reduce artifactual coupling 
[ConcatenatedCifti] = pfm_regress_adjacent_cortex(ConcatenatedCifti,[half_dir '/DistanceMatrix.mat'],20);

% Write out the concatenated CIFTI file
concat_file=[half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_fsLR.dtseries.nii'];
disp(concat_file)
ft_write_cifti_mod(concat_file, ConcatenatedCifti);


%% ---- Step 3: Apply spatial smoothing.
% Define a range of gaussian smoothing kernels (in sigma)
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


%% ---- Step 4: Run infomap.
% Load your concatenated smoothed_file. CHOOSE the smoothing kernel you want to use.
ConcatenatedCifti = ft_read_cifti_mod([half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii']);
DistanceMatrix = [half_dir '/DistanceMatrix.mat']; % can be path to file
DistanceCutoff = 10; % in mm; usually between 10 to 30 mm works well.
GraphDensities = flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]); % 
NumberReps = 50; % number of times infomap is run;
BadVertices = []; % optional, but you could include regions to ignore, if you know there is bad signal there.
Structures = {'CORTEX_LEFT','CEREBELLUM_LEFT','ACCUMBENS_LEFT','CAUDATE_LEFT','PALLIDUM_LEFT','PUTAMEN_LEFT','THALAMUS_LEFT','HIPPOCAMPUS_LEFT','AMYGDALA_LEFT','ACCUMBENS_LEFT','CORTEX_RIGHT','CEREBELLUM_RIGHT','ACCUMBENS_RIGHT','CAUDATE_RIGHT','PALLIDUM_RIGHT','PUTAMEN_RIGHT','THALAMUS_RIGHT','HIPPOCAMPUS_RIGHT','AMYGDALA_RIGHT','ACCUMBENS_RIGHT'};

disp('Starting infomap.')
tic;
pfm_infomap(ConcatenatedCifti,DistanceMatrix,half_dir,GraphDensities,NumberReps,DistanceCutoff,BadVertices,Structures,nWorkers,InfoMapBinary);
elapsed_minutes = toc / 60;
disp(['Elapsed time: ', num2str(elapsed_minutes, '%.2f'), ' minutes'])

% Remove some intermediate files (optional)
system(['rm ' half_dir '/*.net']);
system(['rm ' half_dir '/*.clu']);
system(['rm ' half_dir '/*Log*']);

% Perform spatial filtering
Input = [half_dir '/Bipartite_PhysicalCommunities.dtseries.nii'];
Output = 'Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii';
MinSize = 50; % in mm^2
disp('Performing spatial filtering.')
pfm_spatial_filtering(Input,half_dir,Output,MidthickSurfs,MinSize,WorkbenchBinary);


%% ---- Step 5: Algorithmic assignment of network identities to infomap communities.
load('priors.mat'); % FOR THIS WE WILL LATER NEED TO ADAPT WITH THE NETWORKS WE WANT
Ic = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii']);
Output = 'Bipartite_PhysicalCommunities+AlgorithmicLabeling';
Column = 6; % column 6, representing graph density 0.01% in this example.

% Run the network identification algorithm;
disp('Identifying networks.')
pfm_identify_networks(ConcatenatedCifti,Ic,MidthickSurfs,Column,Priors,Output,half_dir,WorkbenchBinary);


%{
%%---- Step 6: Review algorithmic network assignments, optionally adjust labels manually if needed.

% OPTIONAL: update network assignments according to manual decisions;
XLS = [half_dir '/Bipartite_PhysicalCommunities+AlgorithmicLabeling_NetworkLabels+ManualDecisions.xls']; 
Output = 'Bipartite_PhysicalCommunities+FinalLabeling';
pfm_parse_manual_decisions(Ic,Column,MidthickSurfs,Priors,XLS,Output,half_dir,WorkbenchBinary);


%% ---- Step 7: Calculate size of each functional brain network
FunctionalNetworks = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+FinalLabeling.dlabel.nii']);
VA = ft_read_cifti_mod([Subdir '/fs_LR/fsaverage_LR32k/' subjects{s} '.midthickness_va.32k_fs_LR.dscalar.nii']);
Structures = {'CORTEX_LEFT','CORTEX_RIGHT'}; % in this case, cortex only.

% calculate the size of each functional brain network
NetworkSize = pfm_calculate_network_size(FunctionalNetworks,VA,Structures);

close all; % blank slate
H = figure; % preallocate parent figure
set(H,'position',[1 1 325 400]); hold;

% Unique functional networks
uCi = unique(nonzeros(FunctionalNetworks.data));
for i = 1:length(uCi)
    Tmp = nan(1,length(Priors.NetworkLabels));
    Tmp(i) = NetworkSize(i);
    barh(Tmp,'FaceColor',Priors.NetworkColors(i,:));
    text((NetworkSize(i)+0.1),i,[num2str(NetworkSize(i),3) '%']);
end

yticklabels(Priors.NetworkLabels); 
yticks(1:length(uCi)); ylim([0 21]);
xlim([0 20]); xticks(0:5:20);
set(gca,'fontname','arial','fontsize',10,'TickLength',[0 0],'TickLabelInterpreter','none');
xlabel('% of Cortical Surface');
print(gcf,[PfmDir '/FunctionalNetworkSizes'],'-dpdf');
%} 
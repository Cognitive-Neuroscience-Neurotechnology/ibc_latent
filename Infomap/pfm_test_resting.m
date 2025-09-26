%% A tutorial covering precision functional mapping using an example dataset.
%% This was written by Charles Lynch, PhD, in 2023.
%% This code is part of the PFM-Tutorial repository, available at https://github.com/cjl2007/PFM-Depression

%% ---- Before you begin ---- 

% Add dependencies to Matlab search path & github repo of MSC
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/Infomap/Utilities'));
addpath(genpath('/home/hmueller2/ibc_code/ibc_latent/MSCcodebase/Utilities'))

InfoMapBinary = '/usr/local/bin/infomap';
WorkbenchBinary = '/mnt/workbench/run_wb_command.sh'; % Used in order to use wb_command in apptainer container
nWorkers = 8;

working_dir = '/ptmp/hmueller2/Downloads';

% ---------------------------------

Subject = '05';
Session_1 = '11'; 
Session_2 = '12'; 

% ---------------------------------


%% ---- Step 1: Temporal Concatenation of fMRI data from all sessions.
disp(['---- Step 1: Temporal Concatenation for subject: ' Subject ' ----']);
% tseries_dir = [working_dir '/fmriprep_out/sub-' Subject '/ses-' Session '/postfmriprep/GLM']; 
surface_dir=[working_dir '/fmriprep_out/sub-' Subject]; 

% Only use the two specified resting-state files
files = {
    [surface_dir '/ses-' Session_1 '/postfmriprep_scrubbed/GLM/sub-' Subject '_ses-' Session_1 '_task-RestingState_dir-pa_cleaned.dtseries.nii']
    [surface_dir '/ses-' Session_1 '/postfmriprep_scrubbed/GLM/sub-' Subject '_ses-' Session_1 '_task-RestingState_dir-ap_cleaned.dtseries.nii']
    [surface_dir '/ses-' Session_2 '/postfmriprep_scrubbed/GLM/sub-' Subject '_ses-' Session_2 '_task-RestingState_dir-pa_cleaned.dtseries.nii']
    [surface_dir '/ses-' Session_2 '/postfmriprep_scrubbed/GLM/sub-' Subject '_ses-' Session_2 '_task-RestingState_dir-ap_cleaned.dtseries.nii']
};

ConcatenatedData = [];
used_files = {};
for i = 1:length(files)
    if ~isfile(files{i})
        warning('Resting-state dtseries not found: %s', files{i});
        continue;
    end
    disp(['Loading: ' files{i}])
    try
        Cifti = ft_read_cifti_mod(files{i});
    catch ME
        warning('Failed to read CIFTI: %s\nReason: %s', files{i}, ME.message);
        continue;
    end
    % Ensure shape is (timepoints, grayordinates)
    if size(Cifti.data,2) == 91282
        data_t = Cifti.data;
    elseif size(Cifti.data,1) == 91282
        data_t = Cifti.data';
    else
        warning('Unexpected Cifti.data shape in file: %s. Skipping.', files{i});
        continue;
    end
    if isempty(ConcatenatedData)
        ConcatenatedData = data_t;
    else
        if size(data_t,2) ~= size(ConcatenatedData,2)
            warning('Grayordinate mismatch in file: %s. Skipping.', files{i});
            continue;
        end
        ConcatenatedData = [ConcatenatedData; data_t];
    end
    used_files{end+1} = files{i}; %#ok<AGROW>
end

if isempty(ConcatenatedData)
    error('No usable input CIFTI files were found or readable. Check session paths and filenames.');
end

disp('Loading resting state files completed.'); drawnow;
% Use the last loaded Cifti as template
ConcatenatedCifti = Cifti;
ConcatenatedCifti.data = ConcatenatedData;
disp(['ConcatenatedData shape: ' mat2str(size(ConcatenatedData))]);
disp(['Number of grayordinates in CIFTI: ' num2str(size(ConcatenatedData,2))]);
disp(['Files used (' num2str(numel(used_files)) '):']); disp(used_files');

% Output directories
Subdir = [working_dir '/individual_networks/sub-' Subject];
half_dir = [Subdir '/resting_state'];
mkdir(Subdir); mkdir(half_dir);

% Surface files
MidthickSurfs{1} = [surface_dir '/anat/sub-' Subject '_hemi-L_midthickness.32k_fs_LR.surf.gii'];
MidthickSurfs{2} = [surface_dir '/anat/sub-' Subject '_hemi-R_midthickness.32k_fs_LR.surf.gii'];

% ---- Save inbetween (later overwriten to get rid of coupling)
concat_file = [half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_but_coupled_fsLR.dtseries.nii'];
disp(['Writing concatenated CIFTI: ' concat_file]);
ft_write_cifti_mod(concat_file, ConcatenatedCifti);
% ---- 


%% ---- Step 2: Make a distance matrix.
disp('---- Step 2: Making a distance matrix ----');
tic;
pfm_make_dmat_96k(concat_file,MidthickSurfs,half_dir,nWorkers,WorkbenchBinary);
elapsed_minutes = toc / 60;
disp(['Elapsed time: ', num2str(elapsed_minutes, '%.2f'), ' minutes'])

% Transpose input data if incorrect shape
if size(ConcatenatedCifti.data,1) ~= 91282 && size(ConcatenatedCifti.data,2) == 91282
    ConcatenatedCifti.data = ConcatenatedCifti.data'; % transpose to (91282, timepoints)
end

% Optional: regress adjacent cortical signal from subcortex to reduce artifactual coupling 
disp('Regressing adjacent cortical signal...');
[ConcatenatedCifti] = pfm_regress_adjacent_cortex(ConcatenatedCifti,[half_dir '/DistanceMatrix.mat'],20);

% Write out the concatenated CIFTI file
concat_file=[half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_fsLR.dtseries.nii'];
disp(concat_file)
ft_write_cifti_mod(concat_file, ConcatenatedCifti);


%% ---- Step 3: Apply spatial smoothing.
disp('---- Step 3: Apply spatial smoothing. ----');
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
Kernel = 2.55; % choose one of the kernels from above
disp(['Using smoothing kernel: ' num2str(Kernel)]);
ConcatenatedCifti = ft_read_cifti_mod([half_dir '/sub-' Subject '_all-tasks_concatenated_cleaned_smoothed_' num2str(Kernel) '_fsLR.dtseries.nii']);
DistanceMatrix = [half_dir '/DistanceMatrix.mat']; % can be path to file
DistanceCutoff = 10; % in mm; usually between 10 to 30 mm works well.
GraphDensities = flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]); % 
NumberReps = 50; % number of times infomap is run;
BadVertices = []; % optional, but you could include regions to ignore, if you know there is bad signal there.
Structures = {'CORTEX_LEFT','CEREBELLUM_LEFT','ACCUMBENS_LEFT','CAUDATE_LEFT','PALLIDUM_LEFT','PUTAMEN_LEFT','THALAMUS_LEFT','HIPPOCAMPUS_LEFT','AMYGDALA_LEFT','ACCUMBENS_LEFT','CORTEX_RIGHT','CEREBELLUM_RIGHT','ACCUMBENS_RIGHT','CAUDATE_RIGHT','PALLIDUM_RIGHT','PUTAMEN_RIGHT','THALAMUS_RIGHT','HIPPOCAMPUS_RIGHT','AMYGDALA_RIGHT','ACCUMBENS_RIGHT'};

disp('---- Step 4: Starting infomap.')
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
disp('---- Step 5: Algorithmic assignment of network identities to infomap communities. ----');
load('priors.mat'); % FOR THIS WE WILL LATER NEED TO ADAPT WITH THE NETWORKS WE WANT
Ic = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii']);
Output = 'Bipartite_PhysicalCommunities+AlgorithmicLabeling';
Column = 6; % column 6, representing graph density 0.01% in this example.

% Run the network identification algorithm;
disp('Identifying networks.')
pfm_identify_networks(ConcatenatedCifti,Ic,MidthickSurfs,Column,Priors,Output,half_dir,WorkbenchBinary);


%%---- Step 6: Review algorithmic network assignments, optionally adjust labels manually if needed.
% disp('---- Step 6: Review manual network assignments. ----');
% OPTIONAL: update network assignments according to manual decisions;
% XLS = [half_dir '/Bipartite_PhysicalCommunities+AlgorithmicLabeling_NetworkLabels+ManualDecisions.xls']; 
% Output = 'Bipartite_PhysicalCommunities+FinalLabeling';
% pfm_parse_manual_decisions(Ic,Column,MidthickSurfs,Priors,XLS,Output,half_dir,WorkbenchBinary);



%% ---- Step 7: Calculate size of each functional brain network.
disp('---- Step 7: Calculate size of brain networks. ----');

% Output from Step 6:
% FunctionalNetworks = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+FinalLabeling.dlabel.nii']);

% Output from Step 5 (if Step 6 is skipped):
FunctionalNetworks = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii']);

VA = ft_read_cifti_mod([working_dir '/fmriprep_out/sub-' Subject '/anat/sub-' Subject '.midthickness_va.32k_fs_LR.dscalar.nii']);
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
print(gcf,[half_dir '/FunctionalNetworkSizes'],'-dpdf');

disp(['---- ALL IS DONE FOR SUBJECT ' Subject ' ----']);
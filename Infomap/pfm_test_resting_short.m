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

working_dir = '/ptmp/hmueller2/Downloads';
% ---- DEBUG -----
disp('pfm_test_resting started');

% Which steps to run (now only Step 7)
RUN_STEPS = [7];

% List of subjects to process (edit as needed)
Subjects = {'09','13','14','15'};

%% Step 1: Temporal Concatenation of specific subject and session.

% define subject directory and name;
%Session = '15';

%tseries_dir = [working_dir '/fmriprep_out/sub-' Subject '/ses-' Session '/postfmriprep/GLM']; 


% Output directories
%Subdir = [working_dir '/individual_networks/sub-' Subject];
%half_dir = [Subdir '/resting_state'];
%mkdir(Subdir); mkdir(half_dir);

% define fs_lr_32k midthickness surfaces;
%surface_dir=[working_dir '/fmriprep_out/sub-' Subject]; 

% Surface files
%MidthickSurfs{1} = [surface_dir '/anat/sub-' Subject '_hemi-L_midthickness.32k_fs_LR.surf.gii'];
%MidthickSurfs{2} = [surface_dir '/anat/sub-' Subject '_hemi-R_midthickness.32k_fs_LR.surf.gii'];

% Paths to files already created
%smoothed_file = [half_dir '/sub-' Subject '_ses-' Session '_resting_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii'];
%DistanceMatrix = [half_dir '/DistanceMatrix.mat'];

%% Step 4: Run infomap.
%if ismember(4, RUN_STEPS)
	% load your concatenated resting-state dataset, pick whatever level of spatial smoothing you want
	%ConcatenatedCifti = ft_read_cifti_mod(smoothed_file);

	% define inputs;
	%DistanceMatrix = [half_dir '/DistanceMatrix.mat']; % can be path to file
	%DistanceCutoff = 10; % in mm; usually between 10 to 30 mm works well.
	%GraphDensities = flip([0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05]); % 
	%NumberReps = 5; % Fewer reps for speed
	%BadVertices = []; % optional, but you could include regions to ignore, if you know there is bad signal there.
	%Structures = {'CORTEX_LEFT','CEREBELLUM_LEFT','ACCUMBENS_LEFT','CAUDATE_LEFT','PALLIDUM_LEFT','PUTAMEN_LEFT','THALAMUS_LEFT','HIPPOCAMPUS_LEFT','AMYGDALA_LEFT','ACCUMBENS_LEFT','CORTEX_RIGHT','CEREBELLUM_RIGHT','ACCUMBENS_RIGHT','CAUDATE_RIGHT','PALLIDUM_RIGHT','PUTAMEN_RIGHT','THALAMUS_RIGHT','HIPPOCAMPUS_RIGHT','AMYGDALA_RIGHT','ACCUMBENS_RIGHT'};

	% run infomap
	%disp('starting infomap.')
	%tic;
	%pfm_infomap(ConcatenatedCifti,DistanceMatrix,half_dir,GraphDensities,NumberReps,DistanceCutoff,BadVertices,Structures,nWorkers,InfoMapBinary);
	%elapsedTime = toc;
	%disp(['Elapsed time: ', num2str(elapsedTime), ' seconds'])

	% remove some intermediate files (optional)
	%system(['rm ' half_dir '/*.net']);
	%system(['rm ' half_dir '/*.clu']);
	%system(['rm ' half_dir '/*Log*']);

	% define inputs;
	%Input = [half_dir '/Bipartite_PhysicalCommunities.dtseries.nii'];
	%Output = 'Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii';
	%MinSize = 50; % in mm^2

	% perform spatial filtering
	%disp('spatial filtering.')
	%pfm_spatial_filtering(Input,half_dir,Output,MidthickSurfs,MinSize,WorkbenchBinary);

	%disp(['LH vertices: ' num2str(length(LH.vertices))]);
	%disp(['RH vertices: ' num2str(length(RH.vertices))]);
	%disp(['Total surface vertices: ' num2str(length(LH.vertices) + length(RH.vertices))]);
	%disp(['Length of RefCifti.brainstructure: ' num2str(length(RefCifti.brainstructure))]);
%end

%% Step 5: Algorithmic assignment of network identities to infomap communities.
%if ismember(5, RUN_STEPS)
	% load the priors;
	%load('priors.mat'); % FOR THIS WE WILL LATER NEED TO ADAPT WITH THE NETWORKS WE WANT

	% define inputs;
	%Ic = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii']);
	%Output = 'Bipartite_PhysicalCommunities+AlgorithmicLabeling';
	%Column = 6; % column 6, representing graph density 0.01% in this example.

	% run the network identification algorithm;
	%disp('identifying networks.')
	%pfm_identify_networks(ConcatenatedCifti,Ic,MidthickSurfs,Column,Priors,Output,half_dir,WorkbenchBinary);
%end

%% Step 6: Review algorithmic network assignments, optionally adjust labels manually if needed.
%if ismember(6, RUN_STEPS)
	% define inputs (re-load lightweight dependencies if needed)
	%if ~exist('Ic','var') || isempty(Ic)
		%Ic = ft_read_cifti_mod([half_dir '/Bipartite_PhysicalCommunities+SpatialFiltering.dtseries.nii']);
	%end
	%if ~exist('Column','var') || isempty(Column)
		%Column = 6; % default column if not set
	%end
	%if ~exist('Priors','var') || isempty(Priors)
		%load('priors.mat');
	%end
	% Note: MidthickSurfs must be available from prior steps. If not in workspace,
	% load it if you saved it earlier:
	%   S = load(fullfile(half_dir,'MidthickSurfs.mat')); MidthickSurfs = S.MidthickSurfs;

	%XLS = [half_dir '/Bipartite_PhysicalCommunities+AlgorithmicLabeling_NetworkLabels+ManualDecisions.xls']; 
	%Output = 'Bipartite_PhysicalCommunities+FinalLabeling';

	% OPTIONAL: update network assignments according to manual decisions
	%pfm_parse_manual_decisions(Ic,Column,MidthickSurfs,Priors,XLS,Output,half_dir,WorkbenchBinary);
%end

%% Step 7: Calculate size of each functional brain network
if ismember(7, RUN_STEPS)
    % Ensure Priors available (needed for colors/labels)
    if ~exist('Priors','var') || isempty(Priors)
        load('priors.mat');
    end

    for si = 1:numel(Subjects)
        Subject = Subjects{si};
        Subdir   = [working_dir '/individual_networks/sub-' Subject];
        half_dir = [Subdir '/resting_state'];

        if ~isfolder(half_dir)
            warning('Missing directory for subject %s: %s. Skipping.', Subject, half_dir);
            continue;
        end

        % Prefer FinalLabeling if present, else fall back to AlgorithmicLabeling
        final_file = [half_dir '/Bipartite_PhysicalCommunities+FinalLabeling.dlabel.nii'];
        alg_file   = [half_dir '/Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii'];

        if isfile(final_file)
            fn_file = final_file;
        elseif isfile(alg_file)
            fn_file = alg_file;
            warning('FinalLabeling not found for %s. Using AlgorithmicLabeling.', Subject);
        else
            warning('No labeling file found for subject %s. Skipping.', Subject);
            continue;
        end

        FunctionalNetworks = ft_read_cifti_mod(fn_file);

        % Try primary VA path (as in full script)
        va_file_1 = [working_dir '/fmriprep_out/sub-' Subject '/anat/sub-' Subject '.midthickness_va.32k_fs_LR.dscalar.nii'];
        % Fallback path (older short script style)
        va_file_2 = [Subdir '/fs_LR/fsaverage_LR32k/' Subject '.midthickness_va.32k_fs_LR.dscalar.nii'];

        if isfile(va_file_1)
            VA = ft_read_cifti_mod(va_file_1);
        elseif isfile(va_file_2)
            VA = ft_read_cifti_mod(va_file_2);
        else
            warning('VA file missing for subject %s. Skipping.', Subject);
            continue;
        end

        Structures = {'CORTEX_LEFT','CORTEX_RIGHT'};
        NetworkSize = pfm_calculate_network_size(FunctionalNetworks,VA,Structures);

        close all;
        H = figure;
        set(H,'position',[1 1 325 400]); hold;

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

        out_pdf = [half_dir '/FunctionalNetworkSizes'];
        print(gcf,out_pdf,'-dpdf');
        disp(['Step 7 complete for subject ' Subject '. Output: ' out_pdf '.pdf']);
    end
end

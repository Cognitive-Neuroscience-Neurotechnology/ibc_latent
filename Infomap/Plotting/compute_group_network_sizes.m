function compute_group_network_sizes(subjects_file)
    % Compute group-average functional network sizes from Infomap parcellations
    % Input: subjects_file - path to text file with one subject per line
    
    % -----------------------------
    % 1) Load Priors (network names and colors)
    % -----------------------------
    load('/home/hmueller2/ibc_code/ibc_latent/Infomap/Utilities/priors.mat', 'Priors');
    NetworkNames = Priors.NetworkLabels;
    NetworkColors = Priors.NetworkColors;
    
    % -----------------------------
    % 2) Read subjects from file
    % -----------------------------
    if ~isfile(subjects_file)
        error('Subjects file not found: %s', subjects_file);
    end
    fid = fopen(subjects_file, 'r');
    Subjects = textscan(fid, '%s');
    fclose(fid);
    Subjects = Subjects{1};
    nSubjects = length(Subjects);
    fprintf('Loaded %d subjects from %s\n', nSubjects, subjects_file);
    
    % -----------------------------
    % 3) Directories & constants
    % -----------------------------
    infomap_root = '/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks';
    va_root      = '/ptmp/hmueller2/outputs/preprocessing/fmriprep_out';
    Structures   = {'CORTEX_LEFT','CORTEX_RIGHT'};
    MAX_NETWORKS = length(NetworkNames);
    AllNetworkSizes = nan(nSubjects, MAX_NETWORKS);
    
    % -----------------------------
    % 4) Loop subjects, compute NetworkSize
    % -----------------------------
    for si = 1:nSubjects
        sub = Subjects{si};
        if iscell(sub)
            sub = sub{1};
        end
        sub = char(sub);
        
        fprintf('\n--- (%d/%d) Processing subject: %s ---\n', si, nSubjects, sub);
        
        if startsWith(sub, 'sub-')
            sub_id = sub;
        else
            sub_id = ['sub-' sub];
        end
        
        fn_file = fullfile(infomap_root, sub_id, 'resting_state', ...
                           'Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii');
        va_file = fullfile(va_root, sub_id, 'anat', ...
                           [sub_id '.midthickness_va.32k_fs_LR.dscalar.nii']);
        
        if iscell(fn_file), fn_file = fn_file{1}; end
        if iscell(va_file), va_file = va_file{1}; end
        fn_file = char(fn_file);
        va_file = char(va_file);
        
        if ~isfile(fn_file)
            warning('Missing Infomap file for %s: %s. Skipping.', sub_id, fn_file);
            continue;
        end
        if ~isfile(va_file)
            warning('Missing VA file for %s: %s. Skipping.', sub_id, va_file);
            continue;
        end
        
        try
            NetworkSize = pfm_calculate_network_size(fn_file, va_file, Structures);
        catch ME
            warning('Error computing NetworkSize for %s: %s. Skipping.', sub_id, ME.message);
            continue;
        end
        
        nNet = length(NetworkSize);
        if nNet > MAX_NETWORKS
            warning('Subject %s has %d networks; truncating to %d.', sub_id, nNet, MAX_NETWORKS);
            nNet = MAX_NETWORKS;
        end
        AllNetworkSizes(si, 1:nNet) = NetworkSize(1:nNet);
    end
    
    % -----------------------------
    % 5) Group summary: mean + SEM
    % -----------------------------
    GroupMean = nanmean(AllNetworkSizes, 1);
    GroupSEM  = nanstd(AllNetworkSizes, 0, 1) ./ sqrt(sum(~isnan(AllNetworkSizes),1));
    
    % Find which networks actually exist (non-NaN)
    validNets = find(~isnan(GroupMean));
    
    % -----------------------------
    % 6) Plot horizontal bar chart (matching original style)
    % -----------------------------
    H = figure('Units','pixels','Position',[200 200 900 700]); 
    hold on;
    
    % Plot each network with its specific color
    for i = 1:length(validNets)
        netIdx = validNets(i);
        
        % Create bar for this network only
        Tmp = nan(1, length(validNets));
        Tmp(i) = GroupMean(netIdx);
        barh(i, GroupMean(netIdx), 'FaceColor', NetworkColors(netIdx,:));
        
        % Add error bar
        errorbar(GroupMean(netIdx), i, GroupSEM(netIdx), 'horizontal', 'k.', 'LineWidth', 1.2);
        
        % Add percentage label
        text(GroupMean(netIdx) + 2, i, sprintf('%.2f%%', GroupMean(netIdx)), ...
             'FontSize', 10, 'VerticalAlignment', 'middle');
    end
    
    % Axes and labels
    set(gca, 'YDir', 'reverse');
    ylabel('Functional Network');
    xlabel('% of Cortical Surface');
    title(sprintf('Group-average Functional Network Size (N = %d subjects)', sum(~all(isnan(AllNetworkSizes),2))));
    xlim([0 20]);
    xticks(0:5:20);
    ylim([0.5 length(validNets)+0.5]);
    yticks(1:length(validNets));
    yticklabels(NetworkNames(validNets));
    set(gca, 'FontName', 'Arial', 'FontSize', 10, 'TickLength', [0 0], 'TickLabelInterpreter', 'none');
    grid on;
    
    % -----------------------------
    % 7) Save outputs
    % -----------------------------
    out_mat = fullfile(infomap_root, 'AllNetworkSizes.mat');
    save(out_mat, 'AllNetworkSizes', 'GroupMean', 'GroupSEM', 'Subjects', 'NetworkNames', 'NetworkColors');
    fprintf('Saved group matrix to %s\n', out_mat);
    
    out_pdf = fullfile(infomap_root, 'Group_FunctionalNetworkSizes.pdf');
    print(gcf, out_pdf, '-dpdf', '-bestfit');
    fprintf('Saved figure to %s\n', out_pdf);
    end
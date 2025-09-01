function [Output] = pfm_regress_adjacent_cortex(Input,DistanceMatrix,Distance)

disp(['size(Input.data): ' mat2str(size(Input.data))]);

% count the number of cortical vertices (should be 59412);
nCorticalVertices = nnz(Input.brainstructure==1) + nnz(Input.brainstructure==2);

% load distance matrix;
D = smartload(DistanceMatrix);
disp(['size(D): ' mat2str(size(D))]);

% --- Sanity check: D and Input.data must match in grayordinates ---
if size(D,1) ~= size(Input.data,1)
    error(['Mismatch: Distance matrix has ' num2str(size(D,1)) ...
           ' rows, but Input.data has ' num2str(size(Input.data,1)) ' grayordinates!']);
end

% index of subcortical voxels
SubcortVoxels = (nCorticalVertices+1):size(D,1);

% --- Sanity check: SubcortVoxels must be within bounds ---
if any(SubcortVoxels > size(Input.data,1))
    error('SubcortVoxels indices out of bounds for Input.data!');
end

% trim to be subcortex x cortex;
d = D(SubcortVoxels,1:nCorticalVertices);

% find all voxels adjacent to cortex;
idx = find(min(d,[],2) <= Distance);
disp(['idx: ' mat2str(idx)]);
clear d % clear intermediate file;

% preallocate;
Output = Input;

% sweep all subcortical voxels nearby gray matter;
for i = 1:length(idx)
    % extract nearby gm signals;
    mask = D(SubcortVoxels(idx(i)),:)<=Distance;
    if any(find(mask) > size(Input.data,1))
        error('Mask selects indices out of bounds for Input.data!');
    end
    nb_gm_ts = Input.data(mask,:); %

    % average; if needed
    if size(nb_gm_ts,1) > 1
        nb_gm_ts = mean(nb_gm_ts);
    end

    % remove (possible) contamination of nearby cortical signals via linear regression
    [~,~,Output.data(SubcortVoxels(idx(i)),:),~,~] = regress(Input.data(SubcortVoxels(idx(i)),:)',[nb_gm_ts' ones(size(Input.data,2),1)]);
end

end

% subfunctions
function out = smartload(matfile)
out = load(matfile);
names = fieldnames(out);
out = eval(['out.' names{1}]);
end
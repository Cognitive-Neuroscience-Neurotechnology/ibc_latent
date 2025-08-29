function pfm_make_dmat(RefCifti,MidthickSurfs,OutDir,nWorkers,WorkbenchBinary)
% cjl2007@med.cornell.edu; 

% start parpool;
pool = parpool('local',nWorkers);

try % make hidden directory
    mkdir([OutDir '/tmp/']);
catch
end

% load
% reference CIFTI
if ischar(RefCifti)
    RefCifti = ft_read_cifti_mod(RefCifti);
end

RefCifti.data=[]; % remove data, not needed

% ----- FIXED NOT TO HAVE HARDCODED 96K 
nGray = size(RefCifti.brainstructure,1);

% load midthickness surfaces 
LH = gifti(MidthickSurfs{1});
RH = gifti(MidthickSurfs{2});

% dynamically compute LH/RH indices so they don't exceed array size
lh_idx = find(RefCifti.brainstructure(1:length(LH.vertices)) > 0 & RefCifti.brainstructure(1:length(LH.vertices)) < 3);
rh_idx = find(RefCifti.brainstructure(length(LH.vertices)+1:length(LH.vertices)+length(RH.vertices)) > 0 & RefCifti.brainstructure(length(LH.vertices)+1:length(LH.vertices)+length(RH.vertices)) < 3);

% --- DEBUG: check grayordinate indexing ---
disp(['Total grayordinates in Cifti: ' num2str(size(RefCifti.brainstructure,1))]);

cortical_indices = find(RefCifti.brainstructure > 0 & RefCifti.brainstructure < 3);
subcortical_indices = find(RefCifti.brainstructure > 2);

disp(['Cortical indices: ' num2str(length(cortical_indices))]);
disp(['Subcortical indices: ' num2str(length(subcortical_indices))]);

disp(['Max cortical index: ' num2str(max(cortical_indices))]);
disp(['Max subcortical index: ' num2str(max(subcortical_indices))]);
% ------------------------------------------ NEWW

% split cortical grayordinates into LH and RH
cortical_indices = find(RefCifti.brainstructure > 0 & RefCifti.brainstructure < 3);

% LH vertices in CIFTI grayordinates
lh_cifti = cortical_indices(cortical_indices <= length(LH.vertices));
lh_map = 1:length(lh_cifti); % indices relative to LH.vertices

% RH vertices in CIFTI grayordinates
rh_cifti = cortical_indices(cortical_indices > length(LH.vertices)) - length(LH.vertices);
rh_map = 1:length(rh_cifti); % indices relative to RH.vertices

% ------------------------------------------ 

% LH geodesic distances
lh_mask = lh_cifti; % LH vertices in full LH surface
lh_map = ismember(1:length(LH.vertices), lh_mask);
lh_map = find(lh_map); % row indices for temp.cdata

LH_verts = lh_mask; % vertices to iterate over
parfor i = 1:length(LH_verts)
    system([WorkbenchBinary ' -surface-geodesic-distance ' MidthickSurfs{1} ' ' num2str(LH_verts(i)-1) ' ' OutDir '/tmp/temp_' num2str(i) '.shape.gii']);
    temp = gifti([OutDir '/tmp/temp_' num2str(i) '.shape.gii']);
    system(['rm ' OutDir '/tmp/temp_' num2str(i) '.shape.gii']);
    lh(:,i) = temp.cdata(lh_map); % safe indexing
end

% RH geodesic distances
rh_mask = rh_cifti; % RH vertices in full RH surface
rh_map = ismember(1:length(RH.vertices), rh_mask);
rh_map = find(rh_map); % row indices for temp.cdata

RH_verts = rh_mask;
parfor i = 1:length(RH_verts)
    system([WorkbenchBinary ' -surface-geodesic-distance ' MidthickSurfs{2} ' ' num2str(RH_verts(i)-1) ' ' OutDir '/tmp/temp_' num2str(i) '.shape.gii']);
    temp = gifti([OutDir '/tmp/temp_' num2str(i) '.shape.gii']);
    system(['rm ' OutDir '/tmp/temp_' num2str(i) '.shape.gii']);
    rh(:,i) = temp.cdata(rh_map); % safe indexing
end

% delete 
% parpool
delete(pool);

% remove temp dir.;
[~,~]=system(['rm -rf ' OutDir '/tmp/']);

% convert to uint8
rh = uint8(rh);

% piece together results (999 = inter-hemispheric)
top = [lh ones(length(lh),length(rh))*999]; % lh & dummy rh
bottom = [ones(length(rh),length(lh))*999 rh]; % dummy lh & rh
D = uint8([top;bottom]); % combine hemispheres; cortical surface only so far 

% save distance matrix;
save([OutDir '/DistanceMatrixCortexOnly'],'D','-v7.3');

% extract coordinates for all cortical vertices 
coords_surf=[LH.vertices; RH.vertices]; % combine hemipsheres 

% map cortical vertices directly without truncating to coords_surf length
surf_indices_incifti = RefCifti.brainstructure > 0 & RefCifti.brainstructure < 3;
coords_surf = coords_surf;  % already matches cortical vertices

coords_subcort = RefCifti.pos(RefCifti.brainstructure>2,:);
coords = [coords_surf; coords_subcort]; % combine 

disp(['Size of D: ' mat2str(size(D))]);
disp(['Size of D2 block: ' mat2str(size(D2(1:nCortical,nCortical+1:nGray)))]);
% compute euclidean distance between all vertices & voxels 
D2 = uint8(pdist2(coords,coords));

% combine distance matrices; geodesic & euclidean  
nCortical = size(D,1);
D = [D ; D2(nCortical+1:nGray,1:nCortical)]; % vertcat
D = [D  D2(1:nCortical,nCortical+1:nGray)]; % horzcat
clear D2;

% save distance matrix;
save([OutDir '/DistanceMatrix'],'D','-v7.3');

% clear 
% distances
clear D;

end
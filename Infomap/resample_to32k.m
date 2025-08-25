% Resample native midthickness to fsLR 32k using subject-specific registration spheres.
% Produces:
%   sub-01_hemi-L_midthickness.32k_fs_LR.surf.gii
%   sub-01_hemi-R_midthickness.32k_fs_LR.surf.gii

clear; clc;

% Configure
WorkbenchBinary = '/home/hmueller2/workbench/bin_linux64/wb_command';
in_dir  = '/ptmp/hmueller2/Downloads/fmriprep_out/sub-01/anat';
out_dir = in_dir;
sub = 'sub-01';

% Sanity checks
assert(exist(WorkbenchBinary,'file')==2, 'wb_command not found: %s', WorkbenchBinary);
assert(isfolder(in_dir), 'Input dir not found: %s', in_dir);
assert(isfolder(out_dir), 'Output dir not found: %s', out_dir);

% Input native midthickness
L_mid = fullfile(in_dir, [sub '_hemi-L_midthickness.surf.gii']);
R_mid = fullfile(in_dir, [sub '_hemi-R_midthickness.surf.gii']);
assert(exist(L_mid,'file')==2 && exist(R_mid,'file')==2, 'Native midthickness not found in %s', in_dir);

% Registration spheres (must be registered to fsaverage: sphere.reg)
% Try common names. If not found, error with guidance.
candsL = {
    fullfile(in_dir, [sub '_hemi-L_sphere.reg.surf.gii'])           % fMRIPrep/ciftify style
};
candsR = {
    fullfile(in_dir, [sub '_hemi-R_sphere.reg.surf.gii'])
};

L_sph_reg = first_existing(candsL);
R_sph_reg = first_existing(candsR);

if isempty(L_sph_reg) || isempty(R_sph_reg)
    error(['Registration spheres not found in %s.\n' ...
           'Expected files like:\n  %s_hemi-L_sphere.reg.surf.gii\n  %s_hemi-R_sphere.reg.surf.gii\n' ...
           'If you only have native spheres, create sphere.reg from FreeSurfer:\n' ...
           '  mris_convert $SUBJECTS_DIR/%s/surf/lh.sphere.reg %s_hemi-L_sphere.reg.surf.gii\n' ...
           '  mris_convert $SUBJECTS_DIR/%s/surf/rh.sphere.reg %s_hemi-R_sphere.reg.surf.gii\n'], ...
           in_dir, sub, sub, sub, fullfile(in_dir, [sub '_hemi-L_sphere.reg.surf.gii']), ...
           sub, fullfile(in_dir, [sub '_hemi-R_sphere.reg.surf.gii']));
end

% Target fsLR-32k spheres (shipped with Workbench)
L_target = '/home/hmueller2/workbench/data/standard_mesh_atlases/resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii';
R_target = '/home/hmueller2/workbench/data/standard_mesh_atlases/resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii';
assert(exist(L_target,'file')==2 && exist(R_target,'file')==2, 'fsLR-32k target spheres not found in Workbench data');

% Outputs
L_out = fullfile(out_dir, [sub '_hemi-L_midthickness.32k_fs_LR.surf.gii']);
R_out = fullfile(out_dir, [sub '_hemi-R_midthickness.32k_fs_LR.surf.gii']);

% Commands (quote paths)
cmdL = sprintf('"%s" -surface-resample "%s" "%s" "%s" BARYCENTRIC "%s"', ...
    WorkbenchBinary, L_mid, L_sph_reg, L_target, L_out);
cmdR = sprintf('"%s" -surface-resample "%s" "%s" "%s" BARYCENTRIC "%s"', ...
    WorkbenchBinary, R_mid, R_sph_reg, R_target, R_out);

fprintf('Resampling L:\n%s\n', cmdL);
[statusL, outL] = system(cmdL); if statusL~=0, error('Left resample failed:\n%s', outL); end

fprintf('Resampling R:\n%s\n', cmdR);
[statusR, outR] = system(cmdR); if statusR~=0, error('Right resample failed:\n%s', outR); end

% Verify vertex counts = 32492 each
[~,infoL] = system(sprintf('"%s" -surface-information "%s"', WorkbenchBinary, L_out));
[~,infoR] = system(sprintf('"%s" -surface-information "%s"', WorkbenchBinary, R_out));
fprintf('\n-- Verification --\n%s\n%s\n', infoL, infoR);
assert(contains(infoL, 'Number of vertices: 32492'), 'Left output is not 32k verts:\n%s', infoL);
assert(contains(infoR, 'Number of vertices: 32492'), 'Right output is not 32k verts:\n%s', infoR);

fprintf('\nOK: Wrote 32k midthickness surfaces:\n  %s\n  %s\n', L_out, R_out);

function p = first_existing(list)
    p = '';
    for i=1:numel(list)
        if exist(list{i}, 'file') == 2
            p = list{i};
            return;
        end
    end
end
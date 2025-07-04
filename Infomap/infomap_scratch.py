import nibabel as nib
import numpy as np

# Working in surface space (fsLR 91k) and using the Glasser atlas
dtseries_path = '/ptmp/hmueller2/Downloads/fmriprep_out/sub-01/ses-15/func/sub-01_ses-15_task-RestingState_dir-ap_space-fsLR_den-91k_bold.dtseries.nii'
glasser_dlabel_path = '/Users/hannahmuller/nyx_mount_ptmp/Downloads/atlas_glasser/glasser_hcp/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii'  # <-- update this path to fsLR 91k Glasser atlas

# Load dtseries (time x grayordinates)
dtseries = nib.load(dtseries_path)
dtseries_data = dtseries.get_fdata()  # shape: (time, grayordinates)

# Load Glasser atlas (labels for each grayordinate)
glasser = nib.load(glasser_dlabel_path)
glasser_labels = glasser.get_fdata().squeeze().astype(int)  # shape: (grayordinates,)

# Check that the number of grayordinates matches between data and atlas
if dtseries_data.shape[1] != glasser_labels.shape[0]:
    raise ValueError(f"Mismatch: dtseries has {dtseries_data.shape[1]} grayordinates, "
                     f"atlas has {glasser_labels.shape[0]} labels. Ensure both are in fsLR 91k space.")

# Get unique region labels (excluding 0, which is background)
region_labels = np.unique(glasser_labels)
region_labels = region_labels[region_labels != 0]

# Extract average time series per region
region_ts = []
for label in region_labels:
    idx = np.where(glasser_labels == label)[0]
    region_ts.append(dtseries_data[:, idx].mean(axis=1))
region_ts = np.array(region_ts)  # shape: (n_regions, time)

# Compute functional connectivity matrix (Pearson correlation)
fc_matrix = np.corrcoef(region_ts)

# Save or use fc_matrix as needed
# np.save('fc_matrix.npy', fc_matrix)

import os
import nibabel as nib
import numpy as np
import pandas as pd

def load_surface_maps(base_dir, subject, session, task, contrast, space, hemisphere):
    file_path = f"{base_dir}/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_task-{task}_dir-ffx_space-{space}_hemi-{hemisphere}_ZMap-{contrast}.gii"
    img = nib.load(file_path)
    data = np.array([darray.data for darray in img.darrays])
    return data

def load_atlas_data(atlas_path):
    labels, ctab, names = read_annot(atlas_path)
    return labels, ctab, names

def load_contrast_data(base_dir, subject, session, task, contrast, space, hemisphere):
    file_path = f"{base_dir}/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_task-{task}_dir-ffx_space-{space}_hemi-{hemisphere}_ZMap-{contrast}.gii"
    img = nib.load(file_path)
    return img.get_fdata()
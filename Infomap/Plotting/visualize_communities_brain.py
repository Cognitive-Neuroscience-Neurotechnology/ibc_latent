"""
Take communities from infomap and plot them on the brains.

"""
import numpy as np
import nibabel as nib

def load_clu(filename, skip_header=9):
    data = np.loadtxt(filename, skiprows=skip_header)
    node_ids = data[:, 0].astype(int)
    community_ids = data[:, 1].astype(int)
    return node_ids, community_ids

clu_file = '/ptmp/hmueller2/Downloads/individual_networks/sub-01/resting_state/Bipartite_Density0.05.clu'
template_cifti = '/ptmp/hmueller2/Downloads/individual_networks/sub-01/resting_state/sub-01_ses-15_resting_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii'
output_cifti = '/ptmp/hmueller2/Downloads/individual_networks/sub-01/resting_state/communities.dscalar.nii'

node_ids, community_ids = load_clu(clu_file)
cifti = nib.load(template_cifti)
n_grayordinates = cifti.get_fdata().shape[1]
data = np.zeros(n_grayordinates, dtype=int)
data[node_ids - 1] = community_ids

# Create a new scalar header
scalar_header = nib.cifti2.Cifti2Header.from_header((1, n_grayordinates), 'scalar')
new_img = nib.Cifti2Image(data[np.newaxis, :], header=scalar_header)
nib.save(new_img, output_cifti)
print(f"Saved: {output_cifti}")
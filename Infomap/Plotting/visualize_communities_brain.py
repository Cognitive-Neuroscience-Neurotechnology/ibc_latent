"""
Take communities from infomap and plot them on the brain like the k-means subnetworks.
"""

# Example usage:
# python3 /home/hmueller2/ibc_code/ibc_latent/Infomap/Plotting/visualize_communities_brain.py --subject 01 --dir /ptmp/hmueller2/2025_ibc_latent/outputs --density 0.05

import numpy as np
import nibabel as nib
import argparse
import os

img = nib.load('/home/Downloads/fmriprep_out/sub-01/anat/sub-01_space-fsLR_den-91k_thickness.dscalar.nii')
print(img.get_fdata().shape)

'''
parser = argparse.ArgumentParser(description='Visualize Infomap communities on the brain.')
parser.add_argument('--subject', help='do not include sub- prefix', required=True)
parser.add_argument('--dir', help='Directory to study Data (until derivatives).', required=True)
parser.add_argument('--density', help='Graph density for .clu file', default='0.05')
args = parser.parse_args()

derivative_dir = args.dir
subject = args.subject
density = args.density

networks_dir = os.path.join(derivative_dir, "individual_networks", f'sub-{subject}', 'resting_state')
output_dir = os.path.join(derivative_dir, "subnetworks", "infomap", f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

clu_file = os.path.join(networks_dir, f'Bipartite_Density{density}.clu')
template_cifti = os.path.join(networks_dir, f'sub-{subject}_ses-15_resting_concatenated_cleaned_smoothed_0.85_fsLR.dtseries.nii')
output_cifti = os.path.join(output_dir, f'communities_density{density}.dscalar.nii')

def load_clu(filename, skip_header=9):
    data = np.loadtxt(filename, skiprows=skip_header)
    node_ids = data[:, 0].astype(int)
    community_ids = data[:, 1].astype(int)
    return node_ids, community_ids

print(f"Loading: {clu_file}")
node_ids, community_ids = load_clu(clu_file)
cifti = nib.load(template_cifti)
n_grayordinates = cifti.get_fdata().shape[1]
data = np.zeros(n_grayordinates, dtype=int)
data[node_ids - 1] = community_ids

# Save as dscalar.nii using the template header
new_img = nib.Cifti2Image(data[np.newaxis, :], header=cifti.header)
nib.save(new_img, output_cifti)
print(f"Saved: {output_cifti}")

'''
"""
Take communities from infomap and plot them on the brain like the k-means subnetworks.
"""

import numpy as np
import sys
import argparse
import os

# If RR_utils is needed, adjust the path as appropriate
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR

parser = argparse.ArgumentParser(description='Extract FPN communities from Infomap results')
parser.add_argument('--subject', help='do not include sub- prefix here, e.g. 01')
parser.add_argument('--dir', help='Base working directory, e.g. /ptmp/hmueller2/Downloads')
args = parser.parse_args()

subject = args.subject.zfill(2)  # Ensure two-digit format
working_dir = args.dir

# Input/output directories and file names (matching pfm_tutorial.m)
subdir = os.path.join(working_dir, 'individual_networks', f'sub-{subject}')
half_dir = os.path.join(subdir, 'resting_state')
networks_file = os.path.join(half_dir, 'Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii')

output_dir = os.path.join(working_dir, 'subnetworks', 'infomap', f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

# Load network data
all_networks = RR.load_data(networks_file)
print("all_networks shape:", all_networks.shape)
print("Unique values in all_networks:", np.unique(all_networks))

# Extract FPN mask
fpn_vector = (all_networks[0] == 9).astype(int)
print("Number of FPN vertices in output:", np.sum(fpn_vector))
fpn_vector = fpn_vector.reshape(1, -1)

print("number of found vertices for this network", np.count_nonzero(fpn_vector))
print("number of communities found", np.unique(all_networks).size - 1)

# Make dscalar template (adjust RR function as needed)
dscalar_template, _ = RR.wb_label_to_roi(networks_file, half_dir, 'Frontoparietal')

# Write to cifti
output_file = os.path.join(output_dir, 'FPN_communities.dscalar.nii')
RR.nib_save(output_file, fpn_vector, dscalar_template)

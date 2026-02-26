"""
Take communities from infomap and plot them on the brain like the k-means subnetworks.
"""

import numpy as np
import sys
import argparse
import os

sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR

parser = argparse.ArgumentParser(description='Extract FPN communities from Infomap results')
parser.add_argument('--subject', help='do not include sub- prefix here, e.g. 01')
parser.add_argument('--dir', help='Base working directory, e.g. /ptmp/hmueller2/2025_ibc_latent/outputs')
args = parser.parse_args()

subject = args.subject.zfill(2)  # Ensure two-digit format
working_dir = args.dir

# Input/output directories and file names (matching pfm_tutorial.m)
subdir = os.path.join(working_dir, 'individual_networks', f'sub-{subject}')
half_dir = os.path.join(subdir, 'resting_state')
output_dir = os.path.join(working_dir, 'subnetworks', 'infomap', f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

# Load InfoMap multi-map dlabel (expect shape: n_maps x n_verts)
communities_dlabel = os.path.join(half_dir, 'Bipartite_PhysicalCommunities+AlgorithmicLabeling_InfoMapCommunities.dlabel.nii')
data = RR.load_data(communities_dlabel)
data = np.asarray(data)
print('InfoMap dlabel shape:', data.shape)

if data.ndim != 2 or data.shape[0] < 2:
    raise RuntimeError('Expected multi-map InfoMap dlabel.')

fpn_label = 9  # numeric code for FPN in these maps
n_maps, n_verts = data.shape
combined = np.zeros(n_verts, dtype=np.int32)
next_id = 1

# Build unique community IDs by scanning maps that are FPN-only (values ⊆ {0, 9})
for i in range(n_maps):
    row = data[i, :]
    u = np.unique(row)
    if set(u.tolist()).issubset({0, fpn_label}) and np.any(row == fpn_label):
        size = int((row == fpn_label).sum())
        combined[row == fpn_label] = next_id
        print(f'- map {i+1}: FPN community -> ID {next_id} (size={size})')
        next_id += 1

print('Number of FPN communities found:', next_id - 1)
print('Combined uniques:', np.unique(combined))

# FORCE SUBCORTEX TO ZERO (cortex-only clustering)
CORTEX_SIZE = 64984
if combined.shape[0] > CORTEX_SIZE:
    n_subcortex_labeled = (combined[CORTEX_SIZE:] > 0).sum()
    if n_subcortex_labeled > 0:
        print(f"Zeroing {n_subcortex_labeled} subcortical vertices (keeping cortex-only)")
        combined[CORTEX_SIZE:] = 0  # Force all subcortex to background
    
    print(f"Final combined stats:")
    print(f"  Cortex (0:{CORTEX_SIZE}): {(combined[:CORTEX_SIZE] > 0).sum()} labeled vertices")
    print(f"  Subcortex ({CORTEX_SIZE}:{combined.shape[0]}): {(combined[CORTEX_SIZE:] > 0).sum()} labeled vertices (should be 0)")

# Save as dscalar using a dscalar template (not a dlabel)
# Use the base networks dlabel to make an FPN ROI dscalar for geometry/axes
networks_dlabel = os.path.join(half_dir, 'Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii')
roi_template, ok = RR.wb_label_to_roi(networks_dlabel, half_dir, 'Frontoparietal')
if not ok:
    raise RuntimeError('Failed to create dscalar ROI template from networks dlabel.')

output_file = os.path.join(output_dir, 'FPN_communities.dscalar.nii')
RR.nib_save(output_file, combined.reshape(1, -1), roi_template)
print(f'Saved: {output_file}')
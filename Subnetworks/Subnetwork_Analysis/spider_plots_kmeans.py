"""
Create spider-plots for each subnetwork to study their FC profiles with other LSNs.
Uses pre-relabeled kmeans results on vertices.
"""

# import packages
import os
import numpy as np
import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# arguments
parser = argparse.ArgumentParser(description='Create spider plots from relabeled kmeans results')
parser.add_argument('--subject', help='do not include sub- prefix here, e.g. 01', required=True)
args = parser.parse_args()

subject = args.subject.zfill(2)                  # Ensure two-digit format, e.g. "01"
working_dir = '/ptmp/hmueller2/Downloads'        # Hardcoded base working directory
sub_str = f"sub-{subject}"                       # "sub-04"

"""
00. set up directories
"""
# Output directory for spider plots
kmeans_dir = os.path.join(working_dir, 'subnetworks', 'kmeans', sub_str)
output_dir = os.path.join(working_dir, 'subnetworks', 'kmeans')
spider_plot_dir = os.path.join(output_dir, "spider_plots")
os.makedirs(spider_plot_dir, exist_ok=True)

# Set ptseries and relabeled kmeans paths
parc_filename = os.path.join(
    working_dir, 'individual_networks', sub_str, 'resting_state', f'{sub_str}_individual_nets_concat.ptseries.nii'
)
relabeled_kmeans_path = os.path.join(
    kmeans_dir, f'{sub_str}_kmeans_on_vertices_relabeled.dtseries.nii'
)

# load large scale network parcellation
all_data_concat = RR.load_data(parc_filename)
print("all_data_concat shape:", all_data_concat.shape)  # should be 21 columns

all_data_concat = np.delete(all_data_concat, [8, -1], axis=1)
print("Removed the FPN and Noise. New shape:", all_data_concat.shape)  # should be 19 columns

# load relabeled kmeans vertex labels and timeseries
relabeled_kmeans = RR.load_data(relabeled_kmeans_path)
print("relabeled_kmeans shape:", relabeled_kmeans.shape)

# Load the actual timeseries data from individual networks
timeseries_data = RR.load_data(parc_filename)
print("timeseries_data shape:", timeseries_data.shape)

# Load vertex-level dtseries for subnetwork timeseries extraction
dtseries_path = os.path.join(
    working_dir, 'individual_networks', sub_str, 'resting_state', f'{sub_str}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii'
)
dtseries_concat = RR.load_data(dtseries_path)
print("dtseries_concat shape:", dtseries_concat.shape)

# network names in ptseries in order (abbreviated)
network_names = [
    "Parietal\nDMN",
    "Anterolateral\nDMN",
    "Dorsolateral\nDMN",
    "Retrosplenial\nDMN",
    "Visual\nLateral",
    "Visual\nDorsal",
    "Visual\nV5",
    "Visual\nV1",
    "DAN",
    "DAN\nII",
    "Language",
    "Salience",
    "Cingulo\nOpercular",
    "Medial\nParietal",
    "Somatomotor\nHand",
    "Somatomotor\nFace",
    "Somatomotor\nFoot",
    "Auditory",
    "Somato\nCognitive\nAction"
]

k_values = [2]  # Using k=2 for relabeled results
corr_matrices = {k: {f'{i}': None for i in range(1, k + 1)} for k in k_values}

# Load colors from label table
label_table_file = os.path.join(working_dir, 'subnetworks', 'label_table_infomap_kmeans.txt')
raw_colors = {}
with open(label_table_file, 'r') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
for ln in lines:
    parts = ln.split()
    if len(parts) == 5 and parts[0].isdigit():
        # format: index R G B A
        idx = parts[0]
        r, g, b, a = map(float, parts[1:])
        raw_colors[idx] = (r / 255.0, g / 255.0, b / 255.0, a)

"""
01. Process each subnetwork
"""

for k in k_values:
    print(f'Processing k={k}')

    # Get current kmeans labels for this k value
    # k=2 is at index 0, k=3 at index 1, etc.
    k_index = k - 2
    current_labels = relabeled_kmeans[k_index, :].astype(int).squeeze()
    print(f"current_labels shape: {current_labels.shape}")
    
    labels = {}
    for i in range(1, k + 1):
        rgba = raw_colors.get(str(i), (0.0, 0.0, 0.0, 1.0))
        labels[str(i)] = (f'Subnetwork {i}', rgba)
    
    print(f"Labels dictionary: {labels}")  # Debug print
    
    subnetwork_ids = list(corr_matrices[k].keys())  # e.g., ['1','2',...]

    for subnetwork_id in subnetwork_ids:
        print(f'Processing subnetwork {subnetwork_id}...')
        
        subnetwork_id_int = int(subnetwork_id)
        
        # Create mask for this subnetwork
        mask = (current_labels == subnetwork_id_int)
        num_vertices = np.count_nonzero(mask)
        print(f"Found {num_vertices} vertices for subnetwork {subnetwork_id}.")
        
        if num_vertices == 0:
            print(f"Warning: No vertices found for subnetwork {subnetwork_id}")
            continue
        
        # Get average timeseries for this subnetwork from vertex-level dtseries
        subnetwork_tseries = dtseries_concat[:, mask]
        average_tseries = np.mean(subnetwork_tseries, axis=1)
        
        print(f"average_tseries shape: {average_tseries.shape}")
        
        """
        02. compute correlation matrix with large scale networks
        """
        print("Computing correlation matrix...")
        corr_matrix = np.corrcoef(all_data_concat.T, average_tseries)

        # extract last row for correlations with subnetwork
        correlations_with_column_vector = corr_matrix[-1, :-1]
        print(f"Correlations shape: {correlations_with_column_vector.shape}")

        # store in dictionary
        corr_matrices[k][subnetwork_id] = correlations_with_column_vector

    """
    03. plot spider plots
    """
    print("Plotting Spider Plots...")

    num_columns = len(network_names)

    RR.spider_plot_hannah(output_dir=spider_plot_dir,
                          subject=sub_str,
                          num_columns=num_columns,
                          corr_matrices=corr_matrices[k],
                          labels=labels,
                          network_names=network_names,
                          filename_base=f'kmeans_{k}',
                          minimal=False)

# Save correlation matrices
filename_pkl = os.path.join(output_dir, f'{sub_str}_corr_matrices.pkl')
RR.pickle_save(filename_pkl, corr_matrices)
print(f"Saved correlation matrices to {filename_pkl}")
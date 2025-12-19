"""
Create spider-plots for each subnetwork to study their FC profiles with other LSNs.
"""

# import packages
import os
import numpy as np
import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import argparse
import sys
import re
import matplotlib.pyplot as plt
import pandas as pd
import csv

# arguments
parser = argparse.ArgumentParser(description='Extract FPN communities from Infomap results')
parser.add_argument('--subject', help='do not include sub- prefix here, e.g. 01', required=True)
args = parser.parse_args()

subject = args.subject.zfill(2)                  # Ensure two-digit format, e.g. "01"
working_dir = '/ptmp/hmueller2/Downloads'        # Hardcoded base working directory
sub_str = f"sub-{subject}"                       # "sub-04"

"""
00. set up directories
"""
# Output (and kmeans) directory
subnetwork_dir = os.path.join(working_dir, 'subnetworks')
kmeans_dir = os.path.join(subnetwork_dir, 'infomap', sub_str)
spider_plot_dir = os.path.join(subnetwork_dir, 'spider_plots_additional_k')
infomap_dir = os.path.join(working_dir, 'individual_networks')
#scatter_plot_dir = os.path.join(subnetwork_dir, 'scatter_plots')
os.makedirs(spider_plot_dir, exist_ok=True)
#os.makedirs(scatter_plot_dir, exist_ok=True)

# Set ptseries and dtseries paths directly
parc_filename = os.path.join(
    infomap_dir, sub_str, 'resting_state', f'{sub_str}_individual_nets_concat.ptseries.nii'
)
dtseries_path = os.path.join(
    infomap_dir, sub_str, 'resting_state', f'{sub_str}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii'
)

# concatenate parcellated sessions
all_data_concat = RR.load_data(parc_filename)
print("all_data_concat shape:", all_data_concat.shape)  # should be 21 columns

all_data_concat = np.delete(all_data_concat, [8, -1], axis=1)
print("Removed the FPN and Noise. New shape:", all_data_concat.shape)  # should be 19 columns

# concatenate original dtseries file for the subnetwork tseries (vertex-level)
dtseries_concat = RR.load_data(dtseries_path)

# FPN mask to mask out of corr matrix (not used below, keep for reference)
# FPN_file = os.path.join(kmeans_dir, 'FPN_communities.dscalar.nii')

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

# kmeans labels (use dlabel if present, else dscalar)
kmeans_dlabel = os.path.join(kmeans_dir, f'{subject}_FPN_infomap_communities_kmeans.dlabel.nii')
kmeans_dscalar = os.path.join(kmeans_dir, f'{subject}_FPN_infomap_communities_kmeans.dscalar.nii')
filename = kmeans_dlabel if os.path.exists(kmeans_dlabel) else kmeans_dscalar
subnetworks = RR.load_data(filename)

k_values = range(3, 6) # k_values = range(2, 10)
corr_matrices = {k: {f'{i}': None for i in range(1, k + 1)} for k in k_values}
len_corrs_col = []

# labels are formatted weird for some reason, let it go just until 9
for k in k_values:

    """
    01. create subnetwork masks
    """

    print(f'Creating masks for k={k}')

    # load subnetworks
    current_sns = subnetworks[k-1, :]

    labels = RR.get_labels(filename, n_map=k-1)

    # now we need to figure out the value corresponding to each subnetwork for this subject
    subnetwork_ids = list(corr_matrices[k].keys())  # e.g., ['1','2',...]

    for i, subnetwork_id in enumerate(subnetwork_ids):
        print(f'Processing subnetwork {subnetwork_id}...')

        match_key = next((key for key, value in labels.items() if value[0] == subnetwork_id), None)
        if match_key is None:
            raise ValueError(f'Could not find subnetwork {subnetwork_id} in the labels.')

        # create mask
        print(f"Found {np.count_nonzero(current_sns == match_key)} vertices for subnetwork {subnetwork_id}.")
        mask = RR.create_mask(current_sns, match_key)

        # get average timeseries for this subnetwork (from vertex-level dtseries)
        subnetwork_tseries = RR.get_network(dtseries_concat, mask, remove_rest=True)
        average_tseries = np.mean(subnetwork_tseries, axis=1)

        """
        02. compute correlation matrix
        """
        print("Computing correlation matrix...")
        corr_matrix = np.corrcoef(all_data_concat.T, average_tseries)

        # extract last row / column for correlations with subnetwork
        correlations_with_column_vector = corr_matrix[-1, :-1]
        print(correlations_with_column_vector.shape)

        # store in dictionary
        corr_matrices[k][subnetwork_id] = correlations_with_column_vector
        len_corrs_col.append(len(correlations_with_column_vector))


    """
    03. Relabeling for k=2 to have consistent DMN/DAN assignment
    """
    # After the k-means subnetwork correlation loop, for k=2
    if k == 2:
        dmn_idx = [0, 1, 2, 3]
        dan_idx = [8, 9]
        dmn_dan_means = []
        for subnetwork_id, corr_vec in corr_matrices[k].items():
            if corr_vec is None:
                continue
            dmn_mean = float(np.nanmean(np.asarray(corr_vec)[dmn_idx]))
            dan_mean = float(np.nanmean(np.asarray(corr_vec)[dan_idx]))
            dmn_dan_means.append((subnetwork_id, dmn_mean, dan_mean))
        if len(dmn_dan_means) == 2:
            # Assign 1 to DMN-like, 2 to DAN-like
            sorted_means = sorted(dmn_dan_means, key=lambda x: x[1], reverse=True)
            assign_map = {sorted_means[0][0]: 1, sorted_means[1][0]: 2}
            print(f"Assignment for k=2: {assign_map} (1=DMN, 2=DAN)")
            # Save assignment to CSV
            assign_csv = os.path.join(spider_plot_dir, f"{sub_str}_k2_assignment.csv")
            with open(assign_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['original_subnetwork_id', 'assigned_label', 'mean_DMN', 'mean_DAN'])
                for orig, dmn, dan in dmn_dan_means:
                    writer.writerow([orig, assign_map[orig], dmn, dan])
            print(f"Saved assignment CSV: {assign_csv}")
            # Relabel current_sns so that 1=DMN-like, 2=DAN-like
            relabeled = np.zeros_like(current_sns)
            for cid, newlab in assign_map.items():
                relabeled[current_sns == int(cid)] = newlab
            current_sns = relabeled
            print(f"Applied relabeling: unique labels after remap: {np.unique(current_sns)}")

            # Reorder corr_matrices so that key '1' is DMN-like and '2' is DAN-like for plotting
            try:
                orig_for_1 = next(orig for orig, lab in assign_map.items() if lab == 1)
                orig_for_2 = next(orig for orig, lab in assign_map.items() if lab == 2)
                corr_matrices[k] = {
                    '1': corr_matrices[k][orig_for_1],
                    '2': corr_matrices[k][orig_for_2],
                }
            except StopIteration:
                pass

            # Keep a relabeled copy for downstream scatter plots
            current_sns_relabel = current_sns.copy()

    """
    04. plot spider plots
    """
    print("Plotting Spider Plots...")

    # should be the same for all current_sns so can use this outside of loop
    num_columns = len(correlations_with_column_vector)

    RR.spider_plot_non_interactive(output_dir=spider_plot_dir,
                                   subject=sub_str,
                                   num_columns=num_columns,
                                   corr_matrices=corr_matrices[k],
                                   labels=labels,
                                   network_names=network_names,
                                   filename_base=f'infomap_{k}',
                                   minimal=False)

# Save correlation matrices
filename_pkl = os.path.join(kmeans_dir, f'{sub_str}_corr_matrices.pkl')
RR.pickle_save(filename_pkl, corr_matrices)
print(f"Saved correlation matrices to {filename_pkl}")


"""
05. Scatter plot DMN/DAN connectivity (per-community, colored by subnetwork)

print("Plotting DAN/DMN Connectivity by community...")

# Indices of DMN and DAN networks in network_names
dmn_idx = [0, 1, 2, 3]   # 4 DMN entries
dan_idx = [8, 9]         # 2 DAN entries

# Prefer InfoMap communities dlabel (many IDs); fallback to FPN_communities.dscalar
fpn_comm_src = os.path.join(kmeans_dir, 'FPN_communities.dscalar.nii')
fpn_comm = RR.load_data(fpn_comm_src).astype(int).squeeze()
comm_ids = sorted(int(c) for c in np.unique(fpn_comm) if c > 0)
print(f"[info] found {len(comm_ids)} community IDs (first 12): {comm_ids[:12]}")

for k in k_values:
    # Use relabeled vertex labels if available (ensures colors reflect 1=DMN-like, 2=DAN-like)
    if k == 2 and 'current_sns_relabel' in locals():
        current_sns = current_sns_relabel
    else:
        current_sns = subnetworks[k-1, :]  # cluster labels per vertex (1..k), 0 background
    points, colors, labels_txt = [], [], []
    dmn_dan_means = []

    for cid in comm_ids:
        mask = (fpn_comm == cid)
        if not np.any(mask):
            continue
        # Assign community to subnetwork by majority vote based on (possibly) relabeled current_sns
        lab_vals = current_sns[mask].astype(int)
        lab_vals = lab_vals[lab_vals > 0]
        if lab_vals.size == 0:
            continue
        uniq, cnt = np.unique(lab_vals, return_counts=True)
        cluster = int(uniq[np.argmax(cnt)])  # 1 or 2

        # Average time series for this community
        comm_ts = RR.get_network(dtseries_concat, mask, remove_rest=True)
        avg_ts = np.mean(comm_ts, axis=1)

        # Correlate with LSN ptseries
        corr_matrix = np.corrcoef(all_data_concat.T, avg_ts)
        corr_vec = corr_matrix[-1, :-1]

        dmn_mean = float(np.nanmean(np.asarray(corr_vec)[dmn_idx]))
        dan_mean = float(np.nanmean(np.asarray(corr_vec)[dan_idx]))

        if not (np.isfinite(dmn_mean) and np.isfinite(dan_mean)):
            continue

        points.append((dmn_mean, dan_mean))
        colors.append('C0' if cluster == 1 else 'C1')
        labels_txt.append(str(cid))
        dmn_dan_means.append((cid, dmn_mean, dan_mean))

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.figure(figsize=(5, 4))
        for (x, y), c, t in zip(points, colors, labels_txt):
            plt.scatter(x, y, c=c, s=50, edgecolor='k', linewidths=0.4, alpha=0.9)
            plt.text(x, y, t, fontsize=8, ha='left', va='bottom')
        plt.axvline(0, color='k', lw=0.5, alpha=0.3)
        plt.axhline(0, color='k', lw=0.5, alpha=0.3)
        plt.xlabel("Mean connectivity to DMN (4 nets)")
        plt.ylabel("Mean connectivity to DAN (2 nets)")
        plt.title(f"{sub_str} k={k}: DMN vs DAN per FPN community")
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker='o', color='w', label='Subnetwork 1',
                   markerfacecolor='C0', markeredgecolor='k', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Subnetwork 2',
                   markerfacecolor='C1', markeredgecolor='k', markersize=7),
        ]
        plt.legend(handles=legend_elems, loc='best', frameon=False)
        plt.tight_layout()
        scatter_out = os.path.join(scatter_plot_dir, f"{sub_str}_k{k}_DMN_vs_DAN_scatter_by_community.png")
        plt.savefig(scatter_out, dpi=150)
        plt.close()
        print(f"Saved scatter: {scatter_out}")
"""














"""
Compare subnetwork results from kmeans_on_communities vs kmeans_on_vertices.
Outputs are saved under /subnetworks/comparison/sub-XX/
"""
import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import pearsonr
import argparse
import os
import nibabel as nib
from nibabel.cifti2 import Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

parser = argparse.ArgumentParser(description='Compare two kmeans approaches')
parser.add_argument('--subject', help='subject ID without sub- prefix', required=True)
parser.add_argument('--k', type=int, default=2, help='number of clusters to compare')
args = parser.parse_args()

subject = args.subject.zfill(2)
k = args.k

working_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs'
output_dir = os.path.join(working_dir, 'subnetworks', 'comparison', f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

# Redirect stdout to both console and file
log_file = os.path.join(output_dir, f'comparison_k{k}_log.txt')
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

original_stdout = sys.stdout
log_handle = open(log_file, 'w')
sys.stdout = Tee(sys.stdout, log_handle)

print(f"Processing subject: sub-{subject}")
print(f"Comparing k={k} clusters")

# Load both clustering results
communities_file = os.path.join(working_dir, 'subnetworks', 'infomap', f'sub-{subject}',
                                f'{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii')
vertices_file = os.path.join(working_dir, 'subnetworks', 'kmeans', f'sub-{subject}',
                            f'sub-{subject}_kmeans_on_vertices_relabeled.dlabel.nii')

communities_img = nib.load(communities_file)
vertices_img = nib.load(vertices_file)

communities_data = RR.load_data(communities_file)  # shape: (n_maps, 91282)
vertices_data = RR.load_data(vertices_file)  # shape: (n_maps, 91282)

print(f"Communities data shape: {communities_data.shape}")
print(f"Vertices data shape: {vertices_data.shape}")

# Check if we need to find the right map for k={k}
# For dlabel files, each row might represent different k values or it might be a single k
if communities_data.shape[0] == 1:
    # Single map file - use it directly
    comm_labels = communities_data[0, :64984]
    print(f"Using single communities map (assuming k={k})")
else:
    # Multiple maps - use k-1 index
    if k-1 >= communities_data.shape[0]:
        print(f"ERROR: k={k} requested but only {communities_data.shape[0]} maps available in communities file")
        sys.exit(1)
    comm_labels = communities_data[k-1, :64984]

if vertices_data.shape[0] == 1:
    # Single map file - use it directly
    vert_labels = vertices_data[0, :64984]
    print(f"Using single vertices map (assuming k={k})")
else:
    # Multiple maps - use k-1 index
    if k-1 >= vertices_data.shape[0]:
        print(f"ERROR: k={k} requested but only {vertices_data.shape[0]} maps available in vertices file")
        sys.exit(1)
    vert_labels = vertices_data[k-1, :64984]

# Get FPN mask
FPN_file = os.path.join(working_dir, 'individual_networks', f'sub-{subject}', 
                        'resting_state', 'Frontoparietal_roi.dscalar.nii')
fpn_mask = np.squeeze(RR.load_data(FPN_file))[:64984].astype(bool)

# Restrict to FPN vertices only
comm_fpn = comm_labels[fpn_mask]
vert_fpn = vert_labels[fpn_mask]

# Remove background (0) vertices
valid_mask = (comm_fpn > 0) & (vert_fpn > 0)
comm_valid = comm_fpn[valid_mask]
vert_valid = vert_fpn[valid_mask]

print(f"\n=== Comparing k={k} clustering for sub-{subject} ===")
print(f"Total FPN vertices: {fpn_mask.sum()}")
print(f"Valid labeled vertices: {valid_mask.sum()}")

# 1. OVERLAP ANALYSIS
print("\n--- Vertex Overlap Analysis ---")
contingency = pd.crosstab(comm_valid, vert_valid, 
                          rownames=['Communities'], colnames=['Vertices'])
print("\nContingency table:")
print(contingency)

# Calculate overlap percentages
for i in range(1, k+1):
    for j in range(1, k+1):
        overlap = ((comm_valid == i) & (vert_valid == j)).sum()
        pct_comm = 100 * overlap / (comm_valid == i).sum() if (comm_valid == i).sum() > 0 else 0
        pct_vert = 100 * overlap / (vert_valid == j).sum() if (vert_valid == j).sum() > 0 else 0
        print(f"Communities-{i} ∩ Vertices-{j}: {overlap} verts ({pct_comm:.1f}% of C{i}, {pct_vert:.1f}% of V{j})")

# 2. AGREEMENT METRICS
print("\n--- Agreement Metrics ---")
ari = adjusted_rand_score(comm_valid, vert_valid)
nmi = normalized_mutual_info_score(comm_valid, vert_valid)
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")

# 3. VISUALIZATION: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Vertex Count'})
ax.set_title(f'sub-{subject}: Vertex Assignment Overlap (k={k})\nARI={ari:.3f}, NMI={nmi:.3f}')
ax.set_xlabel('Vertices-based K-means')
ax.set_ylabel('Communities-based K-means')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'overlap_matrix_k{k}.png'), dpi=300)
print(f"\nSaved confusion matrix: {output_dir}/overlap_matrix_k{k}.png")

# 4. CLUSTER SIZE COMPARISON
print("\n--- Cluster Size Comparison ---")
comm_sizes = [(comm_valid == i).sum() for i in range(1, k+1)]
vert_sizes = [(vert_valid == i).sum() for i in range(1, k+1)]

# Create cluster labels (FPNA, FPNB, FPNC, etc.)
cluster_labels = [chr(65 + i) for i in range(k)]  # A, B, C, ...
cluster_names = [f'FPN{label}' for label in cluster_labels]  # FPNA, FPNB, FPNC, ...

size_df = pd.DataFrame({
    'Cluster': cluster_names,
    'Communities-based': comm_sizes,
    'Vertices-based': vert_sizes
})
print(size_df)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(k)
width = 0.35
ax.bar(x - width/2, comm_sizes, width, label='Communities-based', alpha=0.8)
ax.bar(x + width/2, vert_sizes, width, label='Vertices-based', alpha=0.8)
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Vertices')
ax.set_title(f'sub-{subject}: Cluster Sizes (k={k})')
ax.set_xticks(x)
ax.set_xticklabels(cluster_names)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'cluster_sizes_k{k}.png'), dpi=300)

# 5. SPATIAL AGREEMENT MAP
print("\n--- Creating Spatial Agreement Maps ---")

# Create agreement map (1 = agree, 0 = disagree, -1 = not in FPN)
agreement_map_cortex = np.full(64984, -1.0)  # Initialize with -1 (non-FPN)
agreement_map_cortex[fpn_mask] = (comm_fpn == vert_fpn).astype(float)

# Create disagreement type map (which clusters disagree)
disagreement_map_cortex = np.full(64984, 0.0)
fpn_indices = np.where(fpn_mask)[0]
disagree_mask = comm_fpn != vert_fpn
disagreement_map_cortex[fpn_indices[disagree_mask]] = comm_fpn[disagree_mask] * 10 + vert_fpn[disagree_mask]

# Extend to full 91282 grayordinates
agreement_map_full = np.full(91282, -1.0)
agreement_map_full[:64984] = agreement_map_cortex
disagreement_map_full = np.zeros(91282)
disagreement_map_full[:64984] = disagreement_map_cortex

# Save using nibabel directly - copy structure from FPN file
fpn_img = nib.load(FPN_file)

# Create agreement map
agreement_img = nib.Cifti2Image(agreement_map_full.reshape(1, -1), fpn_img.header)
agreement_path = os.path.join(output_dir, f'agreement_map_k{k}.dscalar.nii')
nib.save(agreement_img, agreement_path)
print(f"Saved agreement map: {agreement_path}")

# Create disagreement map
disagreement_img = nib.Cifti2Image(disagreement_map_full.reshape(1, -1), fpn_img.header)
disagreement_path = os.path.join(output_dir, f'disagreement_map_k{k}.dscalar.nii')
nib.save(disagreement_img, disagreement_path)
print(f"Saved disagreement map: {disagreement_path}")

# Calculate agreement statistics
total_fpn = fpn_mask.sum()
agree_count = (comm_fpn == vert_fpn).sum()
disagree_count = (comm_fpn != vert_fpn).sum()
agreement_pct = 100 * agree_count / total_fpn
print(f"Agreement: {agree_count}/{total_fpn} vertices ({agreement_pct:.1f}%)")
print(f"Disagreement: {disagree_count}/{total_fpn} vertices ({100-agreement_pct:.1f}%)")

# 6. CONNECTIVITY PROFILE COMPARISON
print("\n--- Connectivity Profile Comparison ---")

# Load parcellated timeseries
parc_file = os.path.join(working_dir, 'individual_networks', f'sub-{subject}', 
                         'resting_state', f'sub-{subject}_individual_nets_concat.ptseries.nii')
parc_data = RR.load_data(parc_file)  # (timepoints, 21 networks)
parc_data_clean = np.delete(parc_data, [8, -1], axis=1)  # Remove FPN and Noise -> 19 networks

# Load vertex-level timeseries for subnetworks
dtseries_file = os.path.join(working_dir, 'individual_networks', f'sub-{subject}',
                             'resting_state', f'sub-{subject}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii')
dtseries_data_full = RR.load_data(dtseries_file)  # (timepoints, 91282)
dtseries_data = dtseries_data_full[:, :64984]  # Extract only cortex (timepoints, 64984)

# Network names
network_names = ["Parietal DMN", "Anterolateral DMN", "Dorsolateral DMN", "Retrosplenial DMN",
                 "Visual Lateral", "Visual Dorsal", "Visual V5", "Visual V1", "DAN", "DAN II",
                 "Language", "Salience", "Cingulo Opercular", "Medial Parietal",
                 "Somatomotor Hand", "Somatomotor Face", "Somatomotor Foot", "Auditory", "Somato Cognitive Action"]

# Compute connectivity profiles for both approaches
def compute_connectivity_profile(labels, dtseries, parc_data, fpn_mask):
    """Compute correlation of each cluster with 19 LSNs"""
    unique_labels = np.unique(labels[labels > 0])
    profiles = {}
    
    for cluster_id in unique_labels:
        # Get vertices in this cluster
        cluster_mask = np.zeros(64984, dtype=bool)
        fpn_indices = np.where(fpn_mask)[0]
        cluster_in_fpn = labels == cluster_id
        cluster_mask[fpn_indices[cluster_in_fpn]] = True
        
        # Average timeseries across cluster vertices
        cluster_ts = dtseries[:, cluster_mask].mean(axis=1)
        
        # Check for constant timeseries
        if np.std(cluster_ts) == 0:
            print(f"  WARNING: Cluster {cluster_id} has constant timeseries (std=0)")
            profiles[int(cluster_id)] = [np.nan] * parc_data.shape[1]
            continue
        
        # Correlate with each LSN
        correlations = []
        for net_idx in range(parc_data.shape[1]):
            # Check for constant network timeseries
            if np.std(parc_data[:, net_idx]) == 0:
                correlations.append(np.nan)
            else:
                corr, _ = pearsonr(cluster_ts, parc_data[:, net_idx])
                correlations.append(corr)
        
        profiles[int(cluster_id)] = correlations
    
    return profiles

comm_profiles = compute_connectivity_profile(comm_fpn, dtseries_data, parc_data_clean, fpn_mask)
vert_profiles = compute_connectivity_profile(vert_fpn, dtseries_data, parc_data_clean, fpn_mask)

print("\nConnectivity profiles computed for both approaches")

# Compare profiles (handle NaN)
profile_correlations = []
for cluster_id in range(1, k+1):
    if cluster_id in comm_profiles and cluster_id in vert_profiles:
        comm_prof = np.array(comm_profiles[cluster_id])
        vert_prof = np.array(vert_profiles[cluster_id])
        
        # Remove NaN values for correlation
        valid_mask = ~(np.isnan(comm_prof) | np.isnan(vert_prof))
        if valid_mask.sum() < 2:
            print(f"Cluster {cluster_id}: Cannot compute profile correlation (too many NaN)")
            profile_correlations.append(np.nan)
        else:
            corr, _ = pearsonr(comm_prof[valid_mask], vert_prof[valid_mask])
            profile_correlations.append(corr)
            print(f"Cluster {cluster_id}: Profile correlation = {corr:.4f} ({valid_mask.sum()}/19 valid)")

# Visualize connectivity profiles
fig, axes = plt.subplots(k, 2, figsize=(14, 5*k))
if k == 1:
    axes = axes.reshape(1, -1)

for cluster_id in range(1, k+1):
    idx = cluster_id - 1
    cluster_name = cluster_names[idx]
    
    # Communities-based
    if cluster_id in comm_profiles:
        axes[idx, 0].bar(range(19), comm_profiles[cluster_id], alpha=0.7, color='steelblue')
        axes[idx, 0].set_title(f'Communities-based: {cluster_name}')
        axes[idx, 0].set_ylabel('Correlation')
        axes[idx, 0].set_xticks(range(19))
        axes[idx, 0].set_xticklabels(network_names, rotation=45, ha='right', fontsize=8)
        axes[idx, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axes[idx, 0].set_ylim(-0.5, 0.8)
    
    # Vertices-based
    if cluster_id in vert_profiles:
        axes[idx, 1].bar(range(19), vert_profiles[cluster_id], alpha=0.7, color='coral')
        axes[idx, 1].set_title(f'Vertices-based: {cluster_name}')
        axes[idx, 1].set_ylabel('Correlation')
        axes[idx, 1].set_xticks(range(19))
        axes[idx, 1].set_xticklabels(network_names, rotation=45, ha='right', fontsize=8)
        axes[idx, 1].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axes[idx, 1].set_ylim(-0.5, 0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'connectivity_profiles_k{k}.png'), dpi=300)
print(f"\nSaved connectivity profiles: {output_dir}/connectivity_profiles_k{k}.png")

# Overlay comparison plot
fig, axes = plt.subplots(1, k, figsize=(7*k, 5))
if k == 1:
    axes = [axes]

for cluster_id in range(1, k+1):
    idx = cluster_id - 1
    cluster_name = cluster_names[idx]
    if cluster_id in comm_profiles and cluster_id in vert_profiles:
        x = np.arange(19)
        width = 0.35
        axes[idx].bar(x - width/2, comm_profiles[cluster_id], width, label='Communities-based', alpha=0.8, color='steelblue')
        axes[idx].bar(x + width/2, vert_profiles[cluster_id], width, label='Vertices-based', alpha=0.8, color='coral')
        axes[idx].set_title(f'{cluster_name}\n(r={profile_correlations[idx]:.3f})')
        axes[idx].set_ylabel('Correlation with LSN')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(network_names, rotation=45, ha='right', fontsize=8)
        axes[idx].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axes[idx].legend()
        axes[idx].set_ylim(-0.5, 0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'connectivity_profiles_overlay_k{k}.png'), dpi=300)
print(f"Saved overlay profiles: {output_dir}/connectivity_profiles_overlay_k{k}.png")

# Save connectivity profiles to CSV
profile_df = pd.DataFrame({
    'Network': network_names,
    **{f'Comm_{cluster_names[i-1]}': comm_profiles.get(i, [np.nan]*19) for i in range(1, k+1)},
    **{f'Vert_{cluster_names[i-1]}': vert_profiles.get(i, [np.nan]*19) for i in range(1, k+1)}
})
profile_csv = os.path.join(output_dir, f'connectivity_profiles_k{k}.csv')
profile_df.to_csv(profile_csv, index=False)
print(f"Saved connectivity profiles CSV: {profile_csv}")

print(f"\n✓ Comparison complete. Results saved to {output_dir}/")
print(f"Log saved to: {log_file}")

# Restore stdout and close log file
sys.stdout = original_stdout
log_handle.close()


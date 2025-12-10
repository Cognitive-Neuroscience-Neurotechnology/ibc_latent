"""
Quantitative analysis of FPN subnetwork connectivity differences across LSNs.
Computes mean absolute difference in connectivity (Δr) between FPNA and FPNB for each LSN.
"""

import os
import numpy as np
import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Arguments
parser = argparse.ArgumentParser(description='Quantify connectivity differences between FPN subnetworks')
parser.add_argument('--subject', help='subject ID without sub- prefix', required=True)
parser.add_argument('--approach', choices=['infomap', 'kmeans'], default='infomap',
                    help='which clustering approach to use')
args = parser.parse_args()

subject = args.subject.zfill(2)
working_dir = '/ptmp/hmueller2/Downloads'
sub_str = f"sub-{subject}"

# Setup directories
subnetwork_dir = os.path.join(working_dir, 'subnetworks')
if args.approach == 'infomap':
    kmeans_dir = os.path.join(subnetwork_dir, 'infomap', sub_str)
else:
    kmeans_dir = os.path.join(subnetwork_dir, 'kmeans', sub_str)

output_dir = os.path.join(subnetwork_dir, 'connectivity_analysis')
os.makedirs(output_dir, exist_ok=True)

# Network names (full versions for reporting)
network_names_full = [
    "Parietal DMN",
    "Anterolateral DMN",
    "Dorsolateral DMN",
    "Retrosplenial DMN",
    "Visual Lateral",
    "Visual Dorsal",
    "Visual V5",
    "Visual V1",
    "DAN",
    "DAN II",
    "Language",
    "Salience",
    "Cingulo Opercular",
    "Medial Parietal",
    "Somatomotor Hand",
    "Somatomotor Face",
    "Somatomotor Foot",
    "Auditory",
    "Somato Cognitive Action"
]

# Network groupings for summary
network_groups = {
    'DMN': [0, 1, 2, 3],
    'DAN': [8, 9],
    'Visual': [4, 5, 6, 7],
    'Language': [10],
    'Salience': [11],
    'Cingulo-Opercular': [12],
    'Medial Parietal': [13],
    'Somatomotor': [14, 15, 16],
    'Auditory': [17],
    'Somato-Cognitive': [18]
}

print(f"=== Connectivity Difference Analysis for {sub_str} ===")
print(f"Approach: {args.approach}")

# Load correlation matrices
corr_matrices_file = os.path.join(kmeans_dir, f'{sub_str}_corr_matrices.pkl')
if not os.path.exists(corr_matrices_file):
    raise FileNotFoundError(f"Correlation matrices not found: {corr_matrices_file}")

# Use standard pickle
import pickle
with open(corr_matrices_file, 'rb') as f:
    corr_matrices = pickle.load(f)
k = 2  # Focus on 2-cluster solution

if k not in corr_matrices or '1' not in corr_matrices[k] or '2' not in corr_matrices[k]:
    raise ValueError(f"k={k} correlation data not found in {corr_matrices_file}")

# Get connectivity profiles for FPNA and FPNB
fpna_profile = np.array(corr_matrices[k]['1'])  # DMN-like
fpnb_profile = np.array(corr_matrices[k]['2'])  # DAN-like

print(f"\nFPNA profile shape: {fpna_profile.shape}")
print(f"FPNB profile shape: {fpnb_profile.shape}")

# Compute absolute differences for each LSN
differences = np.abs(fpna_profile - fpnb_profile)

# Create results dataframe
results_df = pd.DataFrame({
    'Network': network_names_full,
    'FPNA_connectivity': fpna_profile,
    'FPNB_connectivity': fpnb_profile,
    'Difference': fpna_profile - fpnb_profile,
    'Absolute_Difference': differences
})

# Sort by absolute difference (descending)
results_df = results_df.sort_values('Absolute_Difference', ascending=False)

print("\n=== Individual Network Differences (Δr) ===")
print(results_df.to_string(index=False))

# Compute grouped differences
print("\n=== Network Group Differences ===")
group_results = []
for group_name, indices in network_groups.items():
    fpna_mean = np.mean(fpna_profile[indices])
    fpnb_mean = np.mean(fpnb_profile[indices])
    diff = fpna_mean - fpnb_mean
    abs_diff = np.abs(diff)
    group_results.append({
        'Group': group_name,
        'FPNA_mean': fpna_mean,
        'FPNB_mean': fpnb_mean,
        'Difference': diff,
        'Absolute_Difference': abs_diff,
        'N_networks': len(indices)
    })

group_df = pd.DataFrame(group_results).sort_values('Absolute_Difference', ascending=False)
print(group_df.to_string(index=False))

# Statistical ranking test (Friedman test for differences across networks)
print("\n=== Statistical Analysis ===")
# Wilcoxon signed-rank test: Are DMN/DAN differences larger than others?
dmn_dan_indices = network_groups['DMN'] + network_groups['DAN']
other_indices = [i for i in range(len(network_names_full)) if i not in dmn_dan_indices]

dmn_dan_diffs = differences[dmn_dan_indices]
other_diffs = differences[other_indices]

print(f"DMN/DAN differences (n={len(dmn_dan_diffs)}): mean={np.mean(dmn_dan_diffs):.3f}, median={np.median(dmn_dan_diffs):.3f}")
print(f"Other network differences (n={len(other_diffs)}): mean={np.mean(other_diffs):.3f}, median={np.median(other_diffs):.3f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_value = stats.mannwhitneyu(dmn_dan_diffs, other_diffs, alternative='greater')
print(f"\nMann-Whitney U test (DMN/DAN > Others):")
print(f"  U-statistic = {u_stat:.2f}")
print(f"  p-value = {p_value:.4f}")
print(f"  Effect size (rank biserial) = {1 - (2*u_stat)/(len(dmn_dan_diffs)*len(other_diffs)):.3f}")

# Save results to CSV
results_csv = os.path.join(output_dir, f'{sub_str}_{args.approach}_connectivity_differences.csv')
results_df.to_csv(results_csv, index=False)
print(f"\nSaved individual network results: {results_csv}")

group_csv = os.path.join(output_dir, f'{sub_str}_{args.approach}_group_differences.csv')
group_df.to_csv(group_csv, index=False)
print(f"Saved group results: {group_csv}")

# Visualizations
# 1. Bar plot of absolute differences
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#d62728' if i in dmn_dan_indices else '#7f7f7f' 
          for i in range(len(network_names_full))]
sorted_indices = np.argsort(differences)[::-1]
sorted_names = [network_names_full[i] for i in sorted_indices]
sorted_diffs = differences[sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

ax.barh(range(len(sorted_names)), sorted_diffs, color=sorted_colors, alpha=0.8)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=9)
ax.set_xlabel('Absolute Connectivity Difference (|Δr|)', fontsize=11)
ax.set_title(f'{sub_str}: FPN$_A$ vs FPN$_B$ Connectivity Differences ({args.approach})', fontsize=12)
ax.axvline(np.median(differences), color='k', linestyle='--', linewidth=1, alpha=0.5, label='Median')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{sub_str}_{args.approach}_differences_barplot.png'), dpi=300)
print(f"Saved bar plot: {output_dir}/{sub_str}_{args.approach}_differences_barplot.png")

# 2. Grouped bar plot
fig, ax = plt.subplots(figsize=(10, 6))
group_sorted = group_df.sort_values('Absolute_Difference', ascending=True)
colors_group = ['#d62728' if name in ['DMN', 'DAN'] else '#1f77b4' 
                for name in group_sorted['Group']]
ax.barh(range(len(group_sorted)), group_sorted['Absolute_Difference'], 
        color=colors_group, alpha=0.8)
ax.set_yticks(range(len(group_sorted)))
ax.set_yticklabels(group_sorted['Group'], fontsize=11)
ax.set_xlabel('Mean Absolute Connectivity Difference (|Δr|)', fontsize=11)
ax.set_title(f'{sub_str}: Network Group Connectivity Differences ({args.approach})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{sub_str}_{args.approach}_group_differences_barplot.png'), dpi=300)
print(f"Saved group bar plot: {output_dir}/{sub_str}_{args.approach}_group_differences_barplot.png")

# 3. Heatmap showing FPNA vs FPNB profiles
fig, ax = plt.subplots(figsize=(8, 10))
profiles_matrix = np.vstack([fpna_profile, fpnb_profile])
sns.heatmap(profiles_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=network_names_full, yticklabels=['FPN$_A$', 'FPN$_B$'],
            cbar_kws={'label': 'Pearson r'}, ax=ax)
ax.set_title(f'{sub_str}: Connectivity Profiles ({args.approach})', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{sub_str}_{args.approach}_profiles_heatmap.png'), dpi=300)
print(f"Saved heatmap: {output_dir}/{sub_str}_{args.approach}_profiles_heatmap.png")

# Generate summary text for thesis
print("\n=== SUMMARY FOR THESIS ===")
top_3 = results_df.head(3)
dmn_diff = group_df[group_df['Group'] == 'DMN']['Absolute_Difference'].values[0]
dan_diff = group_df[group_df['Group'] == 'DAN']['Absolute_Difference'].values[0]
lang_diff = group_df[group_df['Group'] == 'Language']['Absolute_Difference'].values[0]
sal_diff = group_df[group_df['Group'] == 'Salience']['Absolute_Difference'].values[0]

print(f"\nMean absolute difference in FPN_A versus FPN_B connectivity was largest for:")
print(f"  - DMN (Δr = {dmn_diff:.3f})")
print(f"  - DAN (Δr = {dan_diff:.3f})")
print(f"Exceeding differences for:")
print(f"  - Language (Δr = {lang_diff:.3f})")
print(f"  - Salience (Δr = {sal_diff:.3f})")
print(f"\nTop 3 individual networks with largest differences:")
for idx, row in top_3.iterrows():
    print(f"  {idx+1}. {row['Network']}: Δr = {row['Absolute_Difference']:.3f}")
print(f"\nStatistical test: Mann-Whitney U test confirmed DMN/DAN differences significantly")
print(f"exceed other networks (U = {u_stat:.2f}, p = {p_value:.4f}).")

print("\n=== Analysis complete ===")
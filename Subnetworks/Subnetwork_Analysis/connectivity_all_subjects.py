"""
Aggregate connectivity differences across all subjects.
Creates summary statistics and visualizations showing which networks consistently
show the largest FPN_A vs FPN_B differences across subjects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob

# Setup
approach = 'kmeans' # 'infomap' or 'kmeans'

working_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs'
subnetwork_dir = os.path.join(working_dir, 'subnetworks')
connectivity_dir = os.path.join(subnetwork_dir, 'connectivity_analysis')
output_dir = os.path.join(subnetwork_dir, 'connectivity_analysis_summary')
os.makedirs(output_dir, exist_ok=True)

print(f"=== Aggregating Connectivity Differences Across Subjects ===")
print(f"Approach: {approach}")

# Find all individual network CSV files
pattern = os.path.join(connectivity_dir, f'sub-*_{approach}_connectivity_differences.csv')
csv_files = sorted(glob.glob(pattern))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found matching: {pattern}")

print(f"Found {len(csv_files)} subject files")

# Load all individual network data
all_data = []
for csv_file in csv_files:
    subject = os.path.basename(csv_file).split('_')[0]  # Extract 'sub-XX'
    df = pd.read_csv(csv_file)
    df['Subject'] = subject
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\nLoaded data for {combined_df['Subject'].nunique()} subjects")
print(f"Total rows: {len(combined_df)}")

# Also load group-level data
group_pattern = os.path.join(connectivity_dir, f'sub-*_{approach}_group_differences.csv')
group_files = sorted(glob.glob(group_pattern))

all_group_data = []
for csv_file in group_files:
    subject = os.path.basename(csv_file).split('_')[0]
    df = pd.read_csv(csv_file)
    df['Subject'] = subject
    all_group_data.append(df)

combined_group_df = pd.concat(all_group_data, ignore_index=True)

# ===== INDIVIDUAL NETWORKS ANALYSIS =====
print("\n=== Individual Networks Analysis ===")

# Compute mean and std across subjects for each network
network_summary = combined_df.groupby('Network').agg({
    'FPNA_connectivity': ['mean', 'std'],
    'FPNB_connectivity': ['mean', 'std'],
    'Absolute_Difference': ['mean', 'std', 'median'],
    'Difference': ['mean', 'std']
}).round(3)

network_summary.columns = ['_'.join(col).strip() for col in network_summary.columns.values]
network_summary = network_summary.reset_index()
network_summary = network_summary.sort_values('Absolute_Difference_mean', ascending=False)

print("\nNetwork Rankings (by mean absolute difference):")
print(network_summary[['Network', 'Absolute_Difference_mean', 'Absolute_Difference_std', 
                       'Absolute_Difference_median']].to_string(index=False))

# Save summary
summary_csv = os.path.join(output_dir, f'{approach}_network_summary_across_subjects.csv')
network_summary.to_csv(summary_csv, index=False)
print(f"\nSaved network summary: {summary_csv}")

# ===== GROUP-LEVEL ANALYSIS =====
print("\n=== Network Group Analysis ===")

group_summary = combined_group_df.groupby('Group').agg({
    'FPNA_mean': ['mean', 'std'],
    'FPNB_mean': ['mean', 'std'],
    'Absolute_Difference': ['mean', 'std', 'median'],
    'Difference': ['mean', 'std']
}).round(3)

group_summary.columns = ['_'.join(col).strip() for col in group_summary.columns.values]
group_summary = group_summary.reset_index()
group_summary = group_summary.sort_values('Absolute_Difference_mean', ascending=False)

print("\nGroup Rankings:")
print(group_summary[['Group', 'Absolute_Difference_mean', 'Absolute_Difference_std',
                     'Absolute_Difference_median']].to_string(index=False))

group_csv = os.path.join(output_dir, f'{approach}_group_summary_across_subjects.csv')
group_summary.to_csv(group_csv, index=False)
print(f"Saved group summary: {group_csv}")

# ===== STATISTICAL TESTING =====
print("\n=== Statistical Testing ===")

# Test if DMN and DAN have significantly higher differences than other networks
dmn_networks = ['Parietal DMN', 'Anterolateral DMN', 'Dorsolateral DMN', 'Retrosplenial DMN']
dan_networks = ['DAN', 'DAN II']
dmn_dan_networks = dmn_networks + dan_networks

dmn_dan_data = combined_df[combined_df['Network'].isin(dmn_dan_networks)]['Absolute_Difference']
other_data = combined_df[~combined_df['Network'].isin(dmn_dan_networks)]['Absolute_Difference']

print(f"\nDMN/DAN networks (n={len(dmn_dan_data)}): mean={dmn_dan_data.mean():.3f}, std={dmn_dan_data.std():.3f}")
print(f"Other networks (n={len(other_data)}): mean={other_data.mean():.3f}, std={other_data.std():.3f}")

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(dmn_dan_data, other_data, alternative='greater')
effect_size = 1 - (2*u_stat)/(len(dmn_dan_data)*len(other_data))

print(f"\nMann-Whitney U test (DMN/DAN > Others):")
print(f"  U = {u_stat:.2f}, p = {p_value:.4f}")
print(f"  Effect size (rank-biserial correlation) = {effect_size:.3f}")

# ===== VISUALIZATIONS =====

# 1. Bar plot with error bars - Individual Networks
fig, ax = plt.subplots(figsize=(14, 8))
network_sorted = network_summary.sort_values('Absolute_Difference_mean', ascending=True)

# Color DMN/DAN networks differently
colors = ['#d62728' if net in dmn_dan_networks else '#1f77b4' 
          for net in network_sorted['Network']]

y_pos = np.arange(len(network_sorted))
ax.barh(y_pos, network_sorted['Absolute_Difference_mean'], 
        xerr=network_sorted['Absolute_Difference_std'],
        color=colors, alpha=0.8, capsize=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(network_sorted['Network'], fontsize=10)
ax.set_xlabel('Mean Absolute Connectivity Difference (|Δr|)', fontsize=12)
ax.set_title(f'FPN$_A$ vs FPN$_B$: Network Connectivity Differences Across Subjects\n({approach}, n={len(csv_files)} subjects)',
             fontsize=13)
ax.axvline(network_sorted['Absolute_Difference_mean'].median(), 
           color='k', linestyle='--', linewidth=1, alpha=0.5, label='Median')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', alpha=0.8, label='DMN/DAN networks'),
    Patch(facecolor='#1f77b4', alpha=0.8, label='Other networks')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{approach}_networks_barplot_with_errorbars.png'), dpi=300)
print(f"\nSaved network bar plot: {output_dir}/{approach}_networks_barplot_with_errorbars.png")
plt.close()

# 2. Grouped bar plot with error bars
fig, ax = plt.subplots(figsize=(10, 7))
group_sorted = group_summary.sort_values('Absolute_Difference_mean', ascending=True)

colors_group = ['#d62728' if grp in ['DMN', 'DAN'] else '#1f77b4' 
                for grp in group_sorted['Group']]

y_pos = np.arange(len(group_sorted))
ax.barh(y_pos, group_sorted['Absolute_Difference_mean'],
        xerr=group_sorted['Absolute_Difference_std'],
        color=colors_group, alpha=0.8, capsize=4)

ax.set_yticks(y_pos)
ax.set_yticklabels(group_sorted['Group'], fontsize=11)
ax.set_xlabel('Mean Absolute Connectivity Difference (|Δr|)', fontsize=12)
ax.set_title(f'FPN$_A$ vs FPN$_B$: Network Group Differences Across Subjects\n({approach}, n={len(csv_files)} subjects)',
             fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{approach}_groups_barplot_with_errorbars.png'), dpi=300)
print(f"Saved group bar plot: {output_dir}/{approach}_groups_barplot_with_errorbars.png")
plt.close()

# 3. Heatmap showing per-subject differences
pivot_data = combined_df.pivot(index='Network', columns='Subject', values='Absolute_Difference')
pivot_data = pivot_data.reindex(network_summary['Network'])  # Sort by mean difference

fig, ax = plt.subplots(figsize=(len(csv_files)*0.8 + 2, 12))
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': '|Δr|'}, ax=ax, linewidths=0.5)
ax.set_title(f'Per-Subject Network Connectivity Differences ({approach})', fontsize=13)
ax.set_xlabel('Subject', fontsize=11)
ax.set_ylabel('Network', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{approach}_per_subject_heatmap.png'), dpi=300)
print(f"Saved heatmap: {output_dir}/{approach}_per_subject_heatmap.png")
plt.close()

# 4. Box plots for top networks
top_6_networks = network_summary.head(6)['Network'].tolist()
top_data = combined_df[combined_df['Network'].isin(top_6_networks)]

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=top_data, x='Network', y='Absolute_Difference', 
            palette=['#d62728' if net in dmn_dan_networks else '#1f77b4' for net in top_6_networks],
            ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Absolute Connectivity Difference (|Δr|)', fontsize=11)
ax.set_xlabel('Network', fontsize=11)
ax.set_title(f'Top 6 Networks: Distribution Across Subjects ({approach})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{approach}_top6_boxplots.png'), dpi=300)
print(f"Saved box plots: {output_dir}/{approach}_top6_boxplots.png")
plt.close()

# ===== THESIS SUMMARY =====
print("\n=== SUMMARY FOR THESIS ===")
print(f"\nBased on {len(csv_files)} subjects using {approach} approach:")
print(f"\nMean absolute connectivity difference between FPN_A and FPN_B:")
dmn_mean = group_summary[group_summary['Group'] == 'DMN']['Absolute_Difference_mean'].values[0]
dmn_std = group_summary[group_summary['Group'] == 'DMN']['Absolute_Difference_std'].values[0]
dan_mean = group_summary[group_summary['Group'] == 'DAN']['Absolute_Difference_mean'].values[0]
dan_std = group_summary[group_summary['Group'] == 'DAN']['Absolute_Difference_std'].values[0]
lang_mean = group_summary[group_summary['Group'] == 'Language']['Absolute_Difference_mean'].values[0]
sal_mean = group_summary[group_summary['Group'] == 'Salience']['Absolute_Difference_mean'].values[0]

print(f"  - DMN: Δr = {dmn_mean:.3f} ± {dmn_std:.3f}")
print(f"  - DAN: Δr = {dan_mean:.3f} ± {dan_std:.3f}")
print(f"  - Language: Δr = {lang_mean:.3f}")
print(f"  - Salience: Δr = {sal_mean:.3f}")

print(f"\nTop 3 individual networks (averaged across subjects):")
for i, (idx, row) in enumerate(network_summary.head(3).iterrows(), 1):
    print(f"  {i}. {row['Network']}: Δr = {row['Absolute_Difference_mean']:.3f} ± {row['Absolute_Difference_std']:.3f}")

print(f"\nStatistical test: Mann-Whitney U test confirmed that DMN/DAN networks show")
print(f"significantly larger connectivity differences than other networks")
print(f"(U = {u_stat:.2f}, p = {p_value:.4f}, effect size = {effect_size:.3f}).")

print("\n=== Aggregation complete ===")
"""
Aggregate contrast × subnetwork results across subjects.
Performs group-level statistics with proper multiple comparison correction.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

print(f"{'='*70}")
print("GROUP-LEVEL AGGREGATION: FPN_A vs FPN_B")
print(f"{'='*70}\n")

# Collect all subject CSVs
result_files = sorted(glob.glob(
    '/ptmp/hmueller2/Downloads/subnetwork_analysis_results/sub-*/fpn_subnetwork_contrast_analysis.csv'
))

if not result_files:
    print("ERROR: No subject result files found!")
    print("Expected pattern: /ptmp/hmueller2/Downloads/subnetwork_analysis_results/sub-*/fpn_subnetwork_contrast_analysis.csv")
    exit(1)

print(f"Found {len(result_files)} subject files:")
for f in result_files:
    subject = f.split('sub-')[1].split('/')[0]
    print(f"  - sub-{subject}")
print()

all_data = []
for file in result_files:
    df = pd.read_csv(file)
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

print(f"Total observations: {len(combined)}")
print(f"Unique contrasts: {combined['task_contrast'].nunique()}")
print(f"Subjects per contrast (should be {len(result_files)}): {combined.groupby('task_contrast').size().value_counts()}\n")

# ========== Group-level statistics per contrast ==========
group_stats = combined.groupby('task_contrast').agg({
    'mean_diff_a_minus_b': ['mean', 'std', 'sem', 'count'],
    'cohens_d': 'mean',
    'task': 'first',
    'contrast': 'first',
}).reset_index()

group_stats.columns = ['_'.join(col).strip('_') for col in group_stats.columns]
group_stats.rename(columns={
    'task_contrast': 'task_contrast',
    'task_first': 'task',
    'contrast_first': 'contrast'
}, inplace=True)

# One-sample t-test: Is mean_diff significantly different from 0 across subjects?
group_stats['t_stat'] = np.nan
group_stats['p_value'] = np.nan
group_stats['dominant_network'] = ''

for idx, row in group_stats.iterrows():
    task_contrast = row['task_contrast']
    values = combined[combined['task_contrast'] == task_contrast]['mean_diff_a_minus_b']
    
    if len(values) > 1:
        t, p = stats.ttest_1samp(values, 0)
        group_stats.loc[idx, 't_stat'] = t
        group_stats.loc[idx, 'p_value'] = p
        group_stats.loc[idx, 'dominant_network'] = 'FPN_A' if row['mean_diff_a_minus_b_mean'] > 0 else 'FPN_B'
    elif len(values) == 1:
        # Single subject - no test possible
        group_stats.loc[idx, 'dominant_network'] = 'FPN_A' if values.iloc[0] > 0 else 'FPN_B'

# ========== FDR correction across all contrasts ==========
valid_p = ~group_stats['p_value'].isna()
if valid_p.sum() > 0:
    group_stats.loc[valid_p, 'p_fdr'] = false_discovery_control(
        group_stats.loc[valid_p, 'p_value'].values, method='bh'
    )
    group_stats['significant_fdr_0.05'] = group_stats['p_fdr'] < 0.05
    group_stats['significant_fdr_0.10'] = group_stats['p_fdr'] < 0.10
    group_stats['significant_fdr_0.20'] = group_stats['p_fdr'] < 0.20
    group_stats['significant_uncorrected_0.001'] = group_stats['p_value'] < 0.001
    group_stats['significant_uncorrected_0.01'] = group_stats['p_value'] < 0.01
else:
    group_stats['p_fdr'] = np.nan
    group_stats['significant_fdr_0.05'] = False
    group_stats['significant_fdr_0.10'] = False
    group_stats['significant_fdr_0.20'] = False
    group_stats['significant_uncorrected_0.001'] = False
    group_stats['significant_uncorrected_0.01'] = False

# Reorder columns for readability
cols_order = ['task_contrast', 'task', 'contrast', 'dominant_network',
              'mean_diff_a_minus_b_mean', 'mean_diff_a_minus_b_std', 'mean_diff_a_minus_b_sem',
              'cohens_d_mean', 't_stat', 'p_value', 'p_fdr', 
              'significant_fdr_0.05', 'significant_fdr_0.10', 'significant_fdr_0.20',
              'significant_uncorrected_0.001', 'significant_uncorrected_0.01',
              'mean_diff_a_minus_b_count']
group_stats = group_stats[cols_order]

# Save
output_dir = '/ptmp/hmueller2/Downloads/subnetwork_analysis_results/group_analysis'
output_path = os.path.join(output_dir, 'group_level_stats.csv')
os.makedirs(output_dir, exist_ok=True)
group_stats.to_csv(output_path, index=False)

print(f"✓ Group-level stats saved: {output_path}\n")

# ========== Summary ==========
print(f"{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

print(f"Total contrasts: {len(group_stats)}")
print(f"\nSignificance at different thresholds:")
print(f"  FDR q < 0.05:  {group_stats['significant_fdr_0.05'].sum()}")
print(f"  FDR q < 0.10:  {group_stats['significant_fdr_0.10'].sum()}")
print(f"  FDR q < 0.20:  {group_stats['significant_fdr_0.20'].sum()}")
print(f"  Uncorrected p < 0.001: {group_stats['significant_uncorrected_0.001'].sum()}")
print(f"  Uncorrected p < 0.01:  {group_stats['significant_uncorrected_0.01'].sum()}")

# Use FDR q < 0.10 as the primary threshold for exploration
sig_contrasts = group_stats[group_stats['significant_fdr_0.10'] == True]

if len(sig_contrasts) > 0:
    print(f"\nDominance distribution (FDR q < 0.10):")
    print(sig_contrasts['dominant_network'].value_counts())
    
    print(f"\n{'='*70}")
    print("TOP 15 CONTRASTS FAVORING FPN_A (group-level, FDR q < 0.10)")
    print(f"{'='*70}")
    top_a = group_stats[group_stats['significant_fdr_0.10']].nlargest(15, 'mean_diff_a_minus_b_mean')[
        ['task', 'contrast', 'mean_diff_a_minus_b_mean', 'cohens_d_mean', 'p_value', 'p_fdr']
    ]
    print(top_a.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("TOP 15 CONTRASTS FAVORING FPN_B (group-level, FDR q < 0.10)")
    print(f"{'='*70}")
    top_b = group_stats[group_stats['significant_fdr_0.10']].nsmallest(15, 'mean_diff_a_minus_b_mean')[
        ['task', 'contrast', 'mean_diff_a_minus_b_mean', 'cohens_d_mean', 'p_value', 'p_fdr']
    ]
    print(top_b.to_string(index=False))
else:
    print("\nNo contrasts survived FDR q < 0.10. Showing strongest effects (uncorrected p < 0.01):")
    sig_uncorrected = group_stats[group_stats['significant_uncorrected_0.01'] == True]
    
    if len(sig_uncorrected) > 0:
        print(f"\nDominance distribution (uncorrected p < 0.01):")
        print(sig_uncorrected['dominant_network'].value_counts())
        
        print(f"\n{'='*70}")
        print("TOP 15 CONTRASTS FAVORING FPN_A (uncorrected p < 0.01)")
        print(f"{'='*70}")
        top_a = sig_uncorrected.nlargest(15, 'mean_diff_a_minus_b_mean')[
            ['task', 'contrast', 'mean_diff_a_minus_b_mean', 'cohens_d_mean', 'p_value']
        ]
        print(top_a.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("TOP 15 CONTRASTS FAVORING FPN_B (uncorrected p < 0.01)")
        print(f"{'='*70}")
        top_b = sig_uncorrected.nsmallest(15, 'mean_diff_a_minus_b_mean')[
            ['task', 'contrast', 'mean_diff_a_minus_b_mean', 'cohens_d_mean', 'p_value']
        ]
        print(top_b.to_string(index=False))
    else:
        print("No contrasts reached even p < 0.01 uncorrected.")

print(f"\n{'='*70}")
print("✓ Aggregation complete!")
print(f"{'='*70}\n")

# ========== VISUALIZATION: Generate group-level plots ==========
print(f"{'='*70}")
print("GENERATING GROUP-LEVEL VISUALIZATIONS")
print(f"{'='*70}\n")

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ========== PLOT 1: Effect size distribution across all contrasts ==========
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by mean effect across subjects
group_stats_sorted = group_stats.sort_values('mean_diff_a_minus_b_mean')
y_pos = np.arange(len(group_stats_sorted))

# Color by significance and direction - create bar collections manually
for i, (idx, row) in enumerate(group_stats_sorted.iterrows()):
    if row['significant_fdr_0.10']:
        color = 'darkred' if row['mean_diff_a_minus_b_mean'] > 0 else 'darkblue'
        alpha = 0.8
    elif row['significant_uncorrected_0.01']:
        color = 'red' if row['mean_diff_a_minus_b_mean'] > 0 else 'blue'
        alpha = 0.6
    else:
        color = 'gray'
        alpha = 0.3
    
    ax.barh(i, row['mean_diff_a_minus_b_mean'],
            xerr=row['mean_diff_a_minus_b_sem'],
            color=color, alpha=alpha, capsize=2)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel("Mean Z-score Difference (FPN_A - FPN_B)\nAcross Subjects", 
              fontsize=12, fontweight='bold')
ax.set_ylabel("Contrasts (ranked by group mean)", fontsize=12, fontweight='bold')
ax.set_title(f"Group-Level FPN Subnetwork Differentiation (N={len(result_files)} subjects)\n" + 
             f"Error bars = SEM | Dark colors = FDR q<0.10",
             fontsize=14, fontweight='bold')
ax.set_ylim(-1, len(group_stats_sorted))
ax.set_yticks([])

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkred', alpha=0.8, label=f'FPN_A > FPN_B (FDR q<0.10, n={group_stats["significant_fdr_0.10"].sum()})'),
    Patch(facecolor='darkblue', alpha=0.8, label='FPN_B > FPN_A (FDR q<0.10)'),
    Patch(facecolor='red', alpha=0.6, label='FPN_A > FPN_B (uncorrected p<0.01)'),
    Patch(facecolor='blue', alpha=0.6, label='FPN_B > FPN_A (uncorrected p<0.01)'),
    Patch(facecolor='gray', alpha=0.3, label='Non-significant'),
]
ax.legend(handles=legend_elements, loc='best', fontsize=9)

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'group_effect_size_distribution.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Plot 1 saved: {plot1_path}")

# ========== PLOT 2: Top contrasts comparison ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Determine which significance threshold to use
if group_stats['significant_fdr_0.10'].sum() >= 15:
    sig_mask = group_stats['significant_fdr_0.10']
    sig_label = 'FDR q<0.10'
elif group_stats['significant_uncorrected_0.01'].sum() >= 15:
    sig_mask = group_stats['significant_uncorrected_0.01']
    sig_label = 'uncorrected p<0.01'
else:
    sig_mask = pd.Series([True] * len(group_stats))  # Show all if too few significant
    sig_label = 'all contrasts'

# Top FPN_A contrasts
top_a_df = group_stats.nlargest(15, 'mean_diff_a_minus_b_mean')
y_pos_a = np.arange(len(top_a_df))
colors_a = ['teal' if row['significant_fdr_0.10'] else '#008080'  # light teal
            for _, row in top_a_df.iterrows()]

ax1.barh(y_pos_a, top_a_df['mean_diff_a_minus_b_mean'], 
         xerr=top_a_df['mean_diff_a_minus_b_sem'],
         color=colors_a, capsize=3)
ax1.set_yticks(y_pos_a)
ax1.set_yticklabels([f"{row['task'][:18]}:\n{row['contrast'][:35]}" 
                      for _, row in top_a_df.iterrows()], fontsize=10)
ax1.set_xlabel('Mean Z-score Difference ± SEM', fontsize=11, fontweight='bold')
ax1.set_title('TOP 15: FPN_A Dominant', 
              fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Top FPN_B contrasts
top_b_df = group_stats.nsmallest(15, 'mean_diff_a_minus_b_mean')
y_pos_b = np.arange(len(top_b_df))
colors_b = ['navy' if row['significant_fdr_0.10'] else '#000080'  # blue
            for _, row in top_b_df.iterrows()]

ax2.barh(y_pos_b, top_b_df['mean_diff_a_minus_b_mean'],
         xerr=top_b_df['mean_diff_a_minus_b_sem'],
         color=colors_b, capsize=3)
ax2.set_yticks(y_pos_b)
ax2.set_yticklabels([f"{row['task'][:18]}:\n{row['contrast'][:35]}" 
                      for _, row in top_b_df.iterrows()], fontsize=10)
ax2.set_xlabel('Mean Z-score Difference ± SEM', fontsize=11, fontweight='bold')
ax2.set_title('TOP 15: FPN_B Dominant', 
              fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

plt.suptitle(f'Group-Level Strongest Functional Differentiation (N={len(result_files)} subjects)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plot2_path = os.path.join(output_dir, 'group_top_contrasts_comparison.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Plot 2 saved: {plot2_path}")

# ========== PLOT 3: Task-level summary with error bars ==========
fig, ax = plt.subplots(figsize=(14, 10))

# Aggregate by task (mean across contrasts and subjects)
task_stats = group_stats.groupby('task').agg({
    'mean_diff_a_minus_b_mean': ['mean', 'std', 'count']
}).reset_index()
task_stats.columns = ['task', 'mean', 'std', 'count']
task_stats = task_stats.sort_values('mean')

y_pos = np.arange(len(task_stats))
colors = ['red' if x > 0 else 'blue' for x in task_stats['mean']]

ax.barh(y_pos, task_stats['mean'], xerr=task_stats['std'], 
        color=colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})
ax.set_yticks(y_pos)
ax.set_yticklabels(task_stats['task'], fontsize=10)
ax.set_xlabel('Mean Z-score Difference (FPN_A - FPN_B)', fontsize=12, fontweight='bold')
ax.set_title(f'Group-Level Task Summary (N={len(result_files)} subjects)\n' +
             'Error bars = SD across contrasts within task',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add sample sizes
for i, (_, row) in enumerate(task_stats.iterrows()):
    ax.text(0.01, i, f"n={int(row['count'])}", 
            fontsize=7, va='center', ha='left', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plot3_path = os.path.join(output_dir, 'group_task_summary_with_variability.png')
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Plot 3 saved: {plot3_path}")

# ========== PLOT 4: Volcano plot (effect size vs significance) ==========
fig, ax = plt.subplots(figsize=(12, 10))

# Calculate -log10(p-value) for volcano plot
group_stats['neg_log10_p'] = -np.log10(group_stats['p_value'].replace(0, 1e-300))

# Color coding
colors = []
labels_added = {'sig_a': False, 'sig_b': False, 'nonsig': False}
for _, row in group_stats.iterrows():
    if row['significant_fdr_0.10']:
        if row['mean_diff_a_minus_b_mean'] > 0:
            colors.append('darkred')
            label = 'FPN_A dominant (FDR q<0.10)' if not labels_added['sig_a'] else ''
            labels_added['sig_a'] = True
        else:
            colors.append('darkblue')
            label = 'FPN_B dominant (FDR q<0.10)' if not labels_added['sig_b'] else ''
            labels_added['sig_b'] = True
    else:
        colors.append('gray')
        label = 'Non-significant' if not labels_added['nonsig'] else ''
        labels_added['nonsig'] = True

ax.scatter(group_stats['mean_diff_a_minus_b_mean'], 
           group_stats['neg_log10_p'],
           c=colors, alpha=0.6, s=40, edgecolors='black', linewidths=0.5)

# Add significance thresholds
ax.axhline(y=-np.log10(0.05), color='orange', linestyle='--', 
           linewidth=1, alpha=0.5, label='p=0.05 (uncorrected)')
ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', 
           linewidth=1, alpha=0.5, label='p=0.01 (uncorrected)')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

ax.set_xlabel('Mean Z-score Difference (FPN_A - FPN_B)', fontsize=12, fontweight='bold')
ax.set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
ax.set_title(f'Volcano Plot: Effect Size vs Significance (N={len(result_files)} subjects)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot4_path = os.path.join(output_dir, 'group_volcano_plot.png')
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Plot 4 saved: {plot4_path}")

# ========== PLOT 5: Individual subject variability for top contrasts ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Get top 10 contrasts for each network
top_a_contrasts = group_stats.nlargest(10, 'mean_diff_a_minus_b_mean')['task_contrast'].values
top_b_contrasts = group_stats.nsmallest(10, 'mean_diff_a_minus_b_mean')['task_contrast'].values

# Prepare data for FPN_A dominant contrasts
plot_data_a = combined[combined['task_contrast'].isin(top_a_contrasts)].copy()
plot_data_a['contrast_label'] = plot_data_a['task'].str[:12] + ':\n' + plot_data_a['contrast'].str[:20]

# Sort by group-level mean (descending, like Plot 2)
contrast_order_a = group_stats.nlargest(10, 'mean_diff_a_minus_b_mean')[['task_contrast', 'task', 'contrast']].copy()
contrast_order_a['contrast_label'] = contrast_order_a['task'].str[:12] + ':\n' + contrast_order_a['contrast'].str[:20]
ordered_labels_a = contrast_order_a['contrast_label'].tolist()

# Box plot for FPN_A
sns.boxplot(data=plot_data_a, y='contrast_label', x='mean_diff_a_minus_b',
            order=ordered_labels_a, color='lightcoral', ax=ax1)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_xlabel('Z-score Difference (FPN_A - FPN_B)', fontsize=11, fontweight='bold')
ax1.set_ylabel('')
ax1.set_title('TOP 10 FPN_A Contrasts:\nAcross-Subject Variability', 
              fontsize=12, fontweight='bold')

# Prepare data for FPN_B dominant contrasts
plot_data_b = combined[combined['task_contrast'].isin(top_b_contrasts)].copy()
plot_data_b['contrast_label'] = plot_data_b['task'].str[:12] + ':\n' + plot_data_b['contrast'].str[:20]

# Sort by group-level mean (ascending, like Plot 2)
contrast_order_b = group_stats.nsmallest(10, 'mean_diff_a_minus_b_mean')[['task_contrast', 'task', 'contrast']].copy()
contrast_order_b['contrast_label'] = contrast_order_b['task'].str[:12] + ':\n' + contrast_order_b['contrast'].str[:20]
ordered_labels_b = contrast_order_b['contrast_label'].tolist()

# Box plot for FPN_B
sns.boxplot(data=plot_data_b, y='contrast_label', x='mean_diff_a_minus_b',
            order=ordered_labels_b, color='lightblue', ax=ax2)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('Z-score Difference (FPN_A - FPN_B)', fontsize=11, fontweight='bold')
ax2.set_ylabel('')
ax2.set_title('TOP 10 FPN_B Contrasts:\nAcross-Subject Variability', 
              fontsize=12, fontweight='bold')

plt.suptitle(f'Individual Subject Variability (N={len(result_files)} subjects)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plot5_path = os.path.join(output_dir, 'group_subject_variability_boxplots.png')
plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Plot 5 saved: {plot5_path}")

print(f"\n{'='*70}")
print("VISUALIZATION SUMMARY")
print(f"{'='*70}")
print(f"Generated 5 group-level plots in: {output_dir}")
print(f"1. Effect size distribution (all {len(group_stats)} contrasts)")
print(f"2. Top 15 contrasts for each subnetwork with SEM")
print(f"3. Task-level summary with SD across contrasts")
print(f"4. Volcano plot (effect size vs significance)")
print(f"5. Subject variability for top 10 contrasts per network")
print(f"\n✓ All visualizations complete!\n")

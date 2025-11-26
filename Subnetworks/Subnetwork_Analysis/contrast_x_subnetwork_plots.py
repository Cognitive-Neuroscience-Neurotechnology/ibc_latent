"""
Create visualizations for contrast x subnetwork analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

results_base = '/ptmp/hmueller2/Downloads/subnetwork_analysis_results'
group_dir = os.path.join(results_base, 'group_analysis')
plot_dir = os.path.join(group_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Load group-level data
group_df = pd.read_csv(os.path.join(group_dir, 'group_level_statistics.csv'))
combined_df = pd.read_csv(os.path.join(group_dir, 'all_subjects_combined.csv'))

# Filter significant contrasts
sig_contrasts = group_df[group_df['group_pval_fdr'] < 0.05].copy()
sig_contrasts = sig_contrasts.sort_values('mean_diff_a_minus_b', ascending=False)

print(f"Plotting {len(sig_contrasts)} significant contrasts...")

# ========== Plot 1: Bar plot of top contrasts ==========
n_top = min(20, len(sig_contrasts))
top_contrasts = pd.concat([
    sig_contrasts.head(10),  # Top 10 for A
    sig_contrasts.tail(10)   # Top 10 for B
])

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(top_contrasts))
colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_contrasts['mean_diff_a_minus_b']]

ax.barh(y_pos, top_contrasts['mean_diff_a_minus_b'], 
        xerr=top_contrasts['mean_diff_se'], color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['task']}/{row['contrast']}" for _, row in top_contrasts.iterrows()], fontsize=9)
ax.set_xlabel('Mean Difference (Subnet A - Subnet B)')
ax.set_title('Top Contrasts Differentiating FPN Subnetworks')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '01_top_contrasts_bar.png'), dpi=300)
plt.close()

# ========== Plot 2: Scatter plot (A vs B) ==========
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(group_df['mean_subnet_a'], group_df['mean_subnet_b'],
                    c=group_df['group_pval_fdr'] < 0.05, cmap='RdYlGn_r',
                    s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.plot([-3, 3], [-3, 3], 'k--', alpha=0.5, label='Unity line')
ax.set_xlabel('Mean Z-score in Subnetwork A')
ax.set_ylabel('Mean Z-score in Subnetwork B')
ax.set_title('Subnetwork Activation Comparison')
ax.legend(['Unity', 'Non-sig', 'Sig (FDR<0.05)'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '02_scatter_a_vs_b.png'), dpi=300)
plt.close()

# ========== Plot 3: Heatmap of mean differences by task ==========
pivot_data = group_df.pivot_table(values='mean_diff_a_minus_b', 
                                   index='contrast', columns='task', 
                                   aggfunc='mean')
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(pivot_data, cmap='RdBu_r', center=0, cbar_kws={'label': 'Mean Diff (A - B)'},
            linewidths=0.5, ax=ax)
ax.set_title('Mean Activation Difference Across Tasks and Contrasts')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '03_heatmap_by_task.png'), dpi=300)
plt.close()

# ========== Plot 4: Violin plot of distribution across subjects ==========
# Select top 5 contrasts favoring each network
top_5_a = sig_contrasts.head(5)['contrast'].tolist()
top_5_b = sig_contrasts.tail(5)['contrast'].tolist()
selected = top_5_a + top_5_b

plot_data = combined_df[combined_df['contrast'].isin(selected)].copy()
plot_data['contrast_short'] = plot_data['contrast'].str[:30]  # Truncate names

fig, ax = plt.subplots(figsize=(14, 8))
sns.violinplot(data=plot_data, x='contrast_short', y='mean_diff_a_minus_b', 
               ax=ax, palette='Set2')
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Contrast')
ax.set_ylabel('Mean Difference (A - B)')
ax.set_title('Distribution of Activation Differences Across Subjects')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '04_violin_subject_distribution.png'), dpi=300)
plt.close()

# ========== Plot 5: Effect size (Cohen's d) distribution ==========
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(group_df['cohens_d_avg'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No effect')
ax.axvline(0.2, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Small effect')
ax.axvline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Medium effect')
ax.axvline(0.8, color='purple', linestyle='--', linewidth=1, alpha=0.7, label='Large effect')
ax.set_xlabel("Cohen's d")
ax.set_ylabel('Frequency')
ax.set_title("Distribution of Effect Sizes (Subnetwork A vs B)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '05_effect_size_distribution.png'), dpi=300)
plt.close()

print(f"✓ Plots saved to: {plot_dir}")